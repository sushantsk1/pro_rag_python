from uuid import uuid4

from conftest import FIXTURES, upload
from fastapi.testclient import TestClient

from server.app import create_app
from server.embedding import LocalEmbedder
from server.errors import RagError
from server.store import LocalStore

ROOT = "/api/v1"


def ask(client, collection_id, **overrides):
    return client.post(
        ROOT + "/queries",
        json={
            "collectionId": collection_id,
            "question": "How long is Aster warranty coverage?",
            "mode": "hybrid",
            **overrides,
        },
    )


def test_lifecycle_retrieval_trace_citations_evaluation_and_deletion(client, collection):
    cid = collection["id"]
    uploaded = upload(client, cid)
    assert uploaded.status_code == 202
    document_id = uploaded.json()["items"][0]["id"]
    doc = client.get(f"{ROOT}/documents/{document_id}").json()
    assert doc["status"] == "ready", doc
    assert "filePath" not in doc
    assert doc["chunkCount"] > 0
    chunks = client.get(f"{ROOT}/documents/{document_id}/chunks").json()["items"]
    assert "24 months" in chunks[0]["text"]
    result = ask(client, cid)
    assert result.status_code == 200, result.text
    result = result.json()
    assert result["answerStatus"] == "answered"
    assert [s["name"] for s in result["trace"]["stages"]] == ["lexical", "semantic", "fusion", "rerank"]
    assert result["trace"]["vectorEngine"] == "faiss"
    assert all(any(s["chunkId"] == c["chunkId"] for s in result["sources"]) for c in result["citations"])
    test_case = {
        "collectionId": cid,
        "question": "How long is Aster warranty coverage?",
        "expectedDocumentIds": [document_id],
        "answerable": True,
    }
    assert client.post(ROOT + "/evaluations/cases", json=test_case).status_code == 201
    runs = [
        client.post(ROOT + "/evaluations/runs", json={"collectionId": cid, "mode": m, "topK": 6}).json()
        for m in ("baseline", "hybrid")
    ]
    for run in runs:
        assert run["recallAtK"] == 1
        assert run["citationValidity"] == 1
    assert runs[0]["corpusFingerprint"] == runs[1]["corpusFingerprint"]
    assert runs[0]["caseSetFingerprint"] == runs[1]["caseSetFingerprint"]
    stats = client.get(ROOT + "/stats").json()
    assert stats["evaluationRunCount"] == 2
    assert stats["readyDocumentCount"] == 1
    assert client.get(ROOT + f"/queries/{result['id']}").status_code == 200
    assert client.get(ROOT + "/queries").json()["items"]
    assert client.delete(f"{ROOT}/documents/{document_id}").status_code == 204
    assert ask(client, cid, generate=False).json()["sources"] == []


def test_duplicate_invalid_and_unsupported_uploads(client, collection):
    cid = collection["id"]
    assert upload(client, cid).status_code == 202
    assert upload(client, cid).status_code == 409
    assert upload(client, cid, "bad.pdf", b"not a PDF").status_code == 400
    assert upload(client, cid, "empty.txt", b"").status_code == 400
    assert upload(client, cid, "unsupported.bin", b"bytes").status_code == 415
    assert client.get(ROOT + "/stats").json()["documentCount"] == 1


def test_all_formats_through_upload_api(client, collection):
    names = [
        "shipping.pdf",
        "returns.docx",
        "launch.pptx",
        "metrics.xlsx",
        "products.csv",
        "policy.txt",
        "onboarding.md",
        "incident.html",
    ]
    response = client.post(
        ROOT + "/documents",
        data={"collectionId": collection["id"]},
        files=[("files", (name, (FIXTURES / name).read_bytes())) for name in names],
    )
    assert response.status_code == 202
    client.app.state.service.wait_for_jobs()
    for doc in response.json()["items"]:
        current = client.get(f"{ROOT}/documents/{doc['id']}").json()
        assert current["status"] == "ready", current


def test_document_filters_and_collection_isolation(client, collection):
    a = upload(client, collection["id"]).json()["items"][0]
    second = client.post(ROOT + "/collections", json={"name": "Other"}).json()
    b = upload(client, second["id"], "other.txt", b"Aster warranty coverage lasts 99 months.").json()[
        "items"
    ][0]
    for retrieval_mode in ("baseline", "hybrid"):
        result = ask(
            client, collection["id"], mode=retrieval_mode, generate=False, documentIds=[a["id"]]
        ).json()
        assert result["sources"]
        assert all(s["documentId"] == a["id"] for s in result["sources"])
        assert ask(client, collection["id"], mode=retrieval_mode, documentIds=[b["id"]]).status_code == 404


def test_unsupported_fact_and_conflicts(client, collection):
    cid = collection["id"]
    upload(client, cid)
    assert ask(client, cid, question="What is the battery chemistry?").json()["answerStatus"] == "abstained"
    conflict_ids = [
        upload(client, cid, name).json()["items"][0]["id"] for name in ("conflict-a.txt", "conflict-b.txt")
    ]
    result = ask(
        client, cid, question="When does the Helix support desk close on weekdays?", documentIds=conflict_ids
    ).json()
    assert result["answerStatus"] == "abstained"
    assert len(result["citations"]) == 2
    assert "18:00" in result["answer"] and "20:00" in result["answer"]


def test_validation_cors_and_pagination(client, collection):
    assert client.post(ROOT + "/collections", json={"name": "bad", "unknown": 1}).status_code == 400
    assert client.post(ROOT + "/collections", content="{").status_code == 400
    assert client.post(ROOT + "/collections", json=[]).status_code == 400
    assert ask(client, collection["id"], topK=1.5).status_code == 400
    assert ask(client, collection["id"], generate="false").status_code == 400
    assert ask(client, collection["id"], documentIds=["invalid"]).status_code == 400
    assert client.get(ROOT + "/collections?limit=0").status_code == 400
    assert client.get(ROOT + "/collections?cursor=broken").status_code == 400
    assert client.get(ROOT + "/health", headers={"Origin": "https://example.com"}).status_code == 403
    allowed = client.options(ROOT + "/queries", headers={"Origin": "http://localhost:5173"})
    assert (
        allowed.status_code == 204
        and allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    )
    assert client.get(ROOT + "/absent").json()["error"]["code"] == "NOT_FOUND"
    client.post(ROOT + "/collections", json={"name": "Second"})
    page = client.get(ROOT + "/collections?limit=1").json()
    assert page["nextCursor"]
    next_page = client.get(ROOT + "/collections", params={"limit": 1, "cursor": page["nextCursor"]}).json()
    assert next_page["items"][0]["id"] != page["items"][0]["id"]


def test_failed_document_retry(client, collection):
    service = client.app.state.service
    actual_parser = service.parse

    def broken(*args, **kwargs):
        raise ValueError("temporary parser failure")

    service.parse = broken
    document_id = upload(client, collection["id"]).json()["items"][0]["id"]
    assert client.get(f"{ROOT}/documents/{document_id}").json()["status"] == "failed"
    assert client.get(f"{ROOT}/documents/{document_id}/chunks").status_code == 409
    service.parse = actual_parser
    assert client.post(f"{ROOT}/documents/{document_id}/retry").status_code == 202
    service.wait_for_jobs()
    assert client.get(f"{ROOT}/documents/{document_id}").json()["status"] == "ready"


def test_unavailable_embeddings_preserve_baseline_and_recover(client, collection):
    service = client.app.state.service
    actual_embedder = service.embedder

    class Unavailable:
        status, identity = "unavailable", actual_embedder.identity

        def embed_query(self, text):
            raise RagError(503, "EMBEDDING_UNAVAILABLE", "Local embedding model unavailable.")

        def embed_documents(self, texts):
            return [self.embed_query(text) for text in texts]

    service.embedder = service.engine.embedder = Unavailable()
    doc_id = upload(client, collection["id"]).json()["items"][0]["id"]
    doc = client.get(f"{ROOT}/documents/{doc_id}").json()
    assert doc["status"] == "ready" and "baseline search remains available" in doc["warning"]
    assert ask(client, collection["id"], mode="baseline", generate=False).json()["sources"]
    assert ask(client, collection["id"], generate=False).status_code == 503
    service.embedder = service.engine.embedder = actual_embedder
    assert ask(client, collection["id"], generate=False).status_code == 200
    assert client.get(f"{ROOT}/documents/{doc_id}").json()["warning"] is None


def test_legacy_sqlite_migration_and_restart(config):
    store = LocalStore(config.data_dir)
    cid, did, chunk_id = str(uuid4()), str(uuid4()), str(uuid4())
    store.put("collection", {"id": cid, "name": "Legacy", "description": ""})
    store.put(
        "document",
        {
            "id": did,
            "collectionId": cid,
            "name": "legacy.txt",
            "status": "ready",
            "format": "txt",
            "chunkCount": 1,
            "sha256": "legacy",
        },
    )
    store.put(
        "chunk",
        {
            "id": chunk_id,
            "collectionId": cid,
            "documentId": did,
            "index": 0,
            "text": "Aster warranty coverage lasts 24 months.",
            "embedding": [0.0] * 384,
            "location": {"type": "line"},
        },
    )
    store.close()
    for attempt in range(2):
        with TestClient(create_app(config)) as client:
            assert (
                ask(client, cid, mode="baseline", generate=False).json()["sources"][0]["chunkId"] == chunk_id
            )
            assert ask(client, cid, generate=False).json()["sources"][0]["chunkId"] == chunk_id
            chunk = client.app.state.service.store.get("chunk", chunk_id)
            assert chunk["embeddingModel"] == LocalEmbedder(config).identity
            assert any(chunk["embedding"])


def test_evaluation_fingerprints_and_unanswerable_cases(client, collection):
    cid = collection["id"]
    did = upload(client, cid).json()["items"][0]["id"]
    client.post(
        ROOT + "/evaluations/cases",
        json={
            "collectionId": cid,
            "question": "How long is Aster warranty coverage?",
            "expectedDocumentIds": [did],
            "answerable": True,
        },
    )

    def run():
        response = client.post(ROOT + "/evaluations/runs", json={"collectionId": cid, "mode": "baseline"})
        assert response.status_code == 201, response.text
        return response.json()

    first = run()
    client.post(
        ROOT + "/evaluations/cases",
        json={
            "collectionId": cid,
            "question": "What is the battery chemistry?",
            "expectedDocumentIds": [],
            "answerable": False,
        },
    )
    second = run()
    assert first["recallAtK"] == second["recallAtK"] == 1
    assert first["mrr"] == second["mrr"]
    assert first["caseSetFingerprint"] != second["caseSetFingerprint"]
    assert first["corpusFingerprint"] == second["corpusFingerprint"]
    upload(client, cid, "incremental.txt")
    third = run()
    assert second["corpusFingerprint"] != third["corpusFingerprint"]


def test_collection_deletion_cleans_related_records(client, collection):
    cid = collection["id"]
    upload(client, cid)
    ask(client, cid, generate=False)
    assert client.delete(f"{ROOT}/collections/{cid}").status_code == 204
    store = client.app.state.service.store
    assert not store.all("document", cid) and not store.all("chunk", cid) and not store.all("query", cid)


def test_documents_with_only_stopwords_remain_searchable_without_crashing(client, collection):
    upload(client, collection["id"], "stopwords.txt", b"the and how are")
    response = ask(client, collection["id"], generate=False)
    assert response.status_code == 200
