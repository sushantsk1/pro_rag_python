import numpy as np
import pytest
from conftest import upload

from server.errors import RagError
from server.provider import source_context

URL = "/api/v1/playground/run"


@pytest.mark.parametrize("chunking", ["fixed", "recursive", "parent-child", "semantic"])
@pytest.mark.parametrize("retrieval", ["keyword", "semantic", "hybrid"])
@pytest.mark.parametrize("embedding", ["local", "tfidf"])
def test_real_pipeline_contract(client, chunking, retrieval, embedding):
    response = client.post(
        URL,
        json={
            "chunking": chunking,
            "overlap": 0,
            "chunkSize": 180,
            "retrieval": retrieval,
            "embedding": embedding,
            "generate": False,
        },
    )
    assert response.status_code == 200, response.text
    run = response.json()
    assert [s["id"] for s in run["stages"]] == ["load", "chunk", "embed", "retrieve", "rerank", "generate"]
    assert all(s["status"] == "complete" for s in run["stages"][:5])
    assert run["stages"][5]["status"] == "skipped"
    chunks = run["stages"][1]["output"]["chunks"]
    vectors = run["stages"][2]["output"]
    assert vectors["mode"] == ("tfidf" if embedding == "tfidf" else "stub")
    assert len(chunks) == len(vectors["vectors"]) == run["summary"]["chunkCount"]
    assert {c["id"] for c in chunks} == {v["chunkId"] for v in vectors["vectors"]}
    for row in vectors["vectors"]:
        assert len(row["values"]) == vectors["dimensions"]
        assert np.linalg.norm(row["values"]) == pytest.approx(1, abs=1e-5)
    chunk_ids = {c["id"] for c in chunks}
    assert all(c["id"] in chunk_ids for c in run["stages"][3]["output"]["candidates"])
    assert run["sources"]
    assert all(s["chunkId"] in chunk_ids for s in run["sources"])
    assert run["summary"]["sourceCount"] == len(run["sources"])


def test_partial_run_does_not_load_embedding_model(client, monkeypatch):
    def unexpected(*args):
        pytest.fail("Partial chunk preview must not initialize embeddings")

    monkeypatch.setattr(client.app.state.service.embedder, "embed_documents", unexpected)
    run = client.post(URL, json={"untilStage": 1}).json()
    assert run["stages"][1]["status"] == "complete"
    assert all(s["status"] == "not_run" for s in run["stages"][2:])
    assert run["summary"]["vectorCount"] == 0


def test_chunk_size_changes_output_and_preserves_offsets(client):
    text = "The warranty lasts two years. Returns require a receipt. " * 12

    def run(size):
        return client.post(
            URL,
            json={
                "source": "text",
                "text": text,
                "chunking": "fixed",
                "chunkSize": size,
                "overlap": 20,
                "untilStage": 1,
            },
        ).json()["stages"][1]["output"]["chunks"]

    small, large = run(120), run(300)
    assert len(small) > len(large)
    assert all(c["text"] == text[c["start"] : c["end"]] for c in small)
    assert small[1]["start"] == small[0]["end"] - 20


def test_parent_context_is_expanded_and_deduplicated(client):
    run = client.post(
        URL,
        json={
            "chunking": "parent-child",
            "chunkSize": 120,
            "overlap": 20,
            "embedding": "tfidf",
            "topK": 10,
        },
    ).json()
    parents = {p["id"]: p for p in run["stages"][1]["output"]["parents"]}
    assert len({s["parentId"] for s in run["sources"]}) == len(run["sources"])
    for source in run["sources"]:
        assert source["snippet"] == parents[source["parentId"]]["text"]
    assert run["stages"][5]["output"]["userPrompt"].endswith(source_context(run["sources"]))


def test_ranking_none_preserves_fusion_order(client):
    request = {"ranking": "none", "embedding": "tfidf"}
    run = client.post(URL, json=request).json()
    ranking = run["stages"][4]["output"]
    assert [c["id"] for c in ranking["before"]] == [c["id"] for c in ranking["after"]]
    request["ranking"] = "heuristic"
    ranking = client.post(URL, json=request).json()["stages"][4]["output"]
    scores = [c["scores"]["rerank"] for c in ranking["after"]]
    assert scores == sorted(scores, reverse=True)


def test_retrieval_only_option_makes_no_generation_or_library_writes(client, monkeypatch):
    service = client.app.state.service

    def unexpected(*args, **kwargs):
        pytest.fail("Explicit retrieval-only runs must not call generation")

    monkeypatch.setattr(service.generator, "generate", unexpected)
    before = [p.name for p in service.store.files_dir.iterdir()]
    assert client.post(URL, json={"embedding": "tfidf", "generate": False}).status_code == 200
    assert service.store.all("collection") == []
    assert service.store.all("query") == []
    assert service.store.all("eval_run") == []
    assert [p.name for p in service.store.files_dir.iterdir()] == before


def test_generator_failure_keeps_inspectable_results(client, monkeypatch):
    def fail(*args, **kwargs):
        raise RagError(502, "NVIDIA_RATE_LIMITED", "Try again later.")

    monkeypatch.setattr(client.app.state.service.generator, "generate", fail)
    response = client.post(URL, json={"generate": True, "embedding": "tfidf"})
    assert response.status_code == 200
    run = response.json()
    assert all(stage["status"] == "complete" for stage in run["stages"][:5])
    assert run["stages"][5]["status"] == "error"
    assert run["stages"][5]["output"]["answerStatus"] == "error"
    assert run["stages"][5]["output"]["error"]["code"] == "NVIDIA_RATE_LIMITED"


def test_generation_prompt_matches_context_sent_to_provider(client, monkeypatch):
    calls = []

    def generate(question, sources, **kwargs):
        calls.append((question, sources))
        return {
            "answer": "An answer [S1]",
            "citations": [{"marker": "S1", "chunkId": sources[0]["chunkId"]}],
            "answerStatus": "answered",
        }

    monkeypatch.setattr(client.app.state.service.generator, "generate", generate)
    run = client.post(URL, json={"generate": True, "embedding": "tfidf"}).json()
    assert len(calls) == 1
    assert run["stages"][5]["output"]["userPrompt"] == (
        f"Question: {calls[0][0]}\n\nEvidence:\n{source_context(calls[0][1])}"
    )
    assert run["stages"][5]["output"]["answerStatus"] == "answered"


@pytest.mark.parametrize(
    "settings",
    [
        {"chunking": "invented"},
        {"embedding": "unknown"},
        {"retrieval": "anything"},
        {"chunkSize": True},
        {"topK": 100},
        {"untilStage": -1},
        {"chunkSize": 120, "overlap": 120},
        {"chunking": "semantic", "overlap": 80},
        {"source": "text", "text": " "},
        {"source": "document"},
        {"generate": "true"},
        {"extra": "field"},
    ],
)
def test_invalid_settings_fail(client, settings):
    assert client.post(URL, json=settings).status_code == 400


def test_source_document_is_scoped_and_never_replaced(client, collection):
    doc = upload(client, collection["id"]).json()["items"][0]
    original = client.app.state.service.store.all("chunk", collection["id"])
    response = client.post(
        URL,
        json={
            "source": "document",
            "documentId": doc["id"],
            "collectionId": collection["id"],
            "embedding": "tfidf",
            "chunkSize": 120,
            "overlap": 10,
        },
    )
    assert response.status_code == 200
    assert response.json()["stages"][0]["output"]["documents"][0]["name"] == doc["name"]
    assert client.app.state.service.store.all("chunk", collection["id"]) == original
    other = client.post("/api/v1/collections", json={"name": "Other collection"}).json()
    assert (
        client.post(
            URL,
            json={
                "source": "document",
                "documentId": doc["id"],
                "collectionId": other["id"],
            },
        ).status_code
        == 404
    )


def test_too_many_chunks_is_bounded(client):
    response = client.post(
        URL,
        json={
            "source": "text",
            "text": "warranty text " * 1000,
            "chunking": "fixed",
            "chunkSize": 120,
            "overlap": 119,
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "PLAYGROUND_LIMIT"


def test_busy_experiments_return_retryable_error(client):
    gate = client.app.state.service.playground.gate
    assert gate.acquire(blocking=False)
    try:
        response = client.post(URL, json={})
        assert response.status_code == 429
        assert response.json()["error"]["code"] == "PLAYGROUND_BUSY"
    finally:
        gate.release()
