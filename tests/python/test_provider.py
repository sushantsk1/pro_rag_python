import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from conftest import upload
from fastapi.testclient import TestClient

from server.app import create_app
from server.provider import AnswerGenerator, conflicting_evidence, parse_reply

SOURCES = [
    {
        "chunkId": "a",
        "documentId": "doc-a",
        "documentName": "policy.txt",
        "location": {"type": "line"},
        "snippet": "Aster warranty coverage lasts 24 months.",
    }
]


@pytest.mark.parametrize(
    "reply", ["Fact [S2]", "Fact without a citation", "[S1]", '{"answer":"Fact", "citations":["S99"]}']
)
def test_invalid_or_missing_citations_abstain(reply):
    assert parse_reply(reply, SOURCES)["answerStatus"] == "abstained"


def test_plain_text_and_json_citations():
    for raw in (
        "Warranty lasts 24 months [S1].",
        '```json\n{"answer":"Warranty lasts 24 months", "citations":["S1"]}\n```',
    ):
        result = parse_reply(raw, SOURCES)
        assert result["answerStatus"] == "answered"
        assert result["citations"] == [{"marker": "S1", "chunkId": "a"}]
        assert "[S1]" in result["answer"]


def test_no_evidence_trace_does_not_claim_a_model_request(config):
    trace = {}
    result = AnswerGenerator(config).generate("What is the candidate name?", [], trace=trace)
    assert result["answerStatus"] == "abstained"
    assert trace["decision"] == "no_evidence"
    assert trace["providerRequestAttempted"] is False
    assert trace["providerResponseReceived"] is False


def test_conflicting_hyphenated_claims():
    sources = [
        SOURCES[0],
        {
            **SOURCES[0],
            "chunkId": "b",
            "documentId": "doc-b",
            "documentName": "legacy.txt",
            "snippet": "Aster warranty coverage is a 12-month warranty.",
        },
    ]
    result = conflicting_evidence("How long is Aster warranty coverage?", sources)
    assert result["answerStatus"] == "abstained" and len(result["citations"]) == 2


@pytest.fixture
def fake_nvidia():
    state = {"status": 200, "delay": 0, "requests": [], "content": "Warranty coverage lasts 24 months [S1]."}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "local-fault-model", "object": "model"}]}).encode())

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append((self.path, payload))
            time.sleep(state["delay"])
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            result = {
                "id": "test",
                "object": "chat.completion",
                "model": "local-fault-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": state["content"]},
                        "finish_reason": "stop",
                    }
                ],
            }
            if state["status"] != 200:
                result = {"error": {"message": "local simulated failure", "status": state["status"]}}
            try:
                self.wfile.write(json.dumps(result).encode())
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", state
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_playground_entity_question_reaches_provider_despite_low_ranking_score(config, fake_nvidia):
    base, state = fake_nvidia
    config.model_mode = "nvidia"
    config.nvidia_api_key = "local-dummy-secret"
    config.nvidia_base_url = base
    config.nvidia_model = "local-fault-model"
    state["content"] = '{"answer":"The candidate is Priya Shah [S1].","citations":["S1"],"abstain":false}'
    app = create_app(config)
    # A real normalized semantic hit with no literal query-token overlap.
    app.state.service.embedder.embed_documents = lambda texts: [[1.0, 0.0] for _ in texts]
    app.state.service.embedder.embed_query = lambda text: [0.2, (1 - 0.2**2) ** 0.5]
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/playground/run",
            json={
                "source": "text",
                "text": "Priya Shah\nSoftware engineer working with Python and SQL.",
                "question": "What is the candidate name?",
                "retrieval": "semantic",
            },
        )
        assert response.status_code == 200
        output = response.json()["stages"][5]["output"]
        assert output["retrievalScore"] == pytest.approx(0.06)
        assert output["answerStatus"] == "answered"
        assert "Priya Shah" in output["answer"]
        assert output["citations"][0]["marker"] == "S1"
        assert output["execution"]["providerRequestAttempted"] is True
        assert output["execution"]["providerResponseReceived"] is True
        assert output["execution"]["decision"] == "answered"
        sent = state["requests"][-1][1]["messages"]
        assert sent[0]["content"] == output["systemPrompt"]
        assert sent[1]["content"] == output["userPrompt"]
        assert "Priya Shah" in sent[1]["content"]


@pytest.mark.parametrize(
    "reply,decision",
    [
        ('{"answer":"Evidence is insufficient.","citations":[],"abstain":true}', "model_abstained"),
        ('{"answer":"A claim [S99]","citations":["S99"],"abstain":false}', "citation_validation_failed"),
    ],
)
def test_playground_distinguishes_model_abstention_from_citation_rejection(
    config, fake_nvidia, reply, decision
):
    base, state = fake_nvidia
    config.model_mode = "nvidia"
    config.nvidia_api_key = "local-dummy-secret"
    config.nvidia_base_url = base
    config.nvidia_model = "local-fault-model"
    state["content"] = reply
    with TestClient(create_app(config)) as client:
        run = client.post("/api/v1/playground/run", json={"embedding": "tfidf"}).json()
        output = run["stages"][5]["output"]
        assert output["answerStatus"] == "abstained"
        assert output["execution"]["providerResponseReceived"] is True
        assert output["execution"]["decision"] == decision


def test_actual_langchain_nvidia_chain_and_faults(config, fake_nvidia):
    base, state = fake_nvidia
    config.model_mode = "nvidia"
    config.nvidia_api_key = "local-dummy-secret"
    config.nvidia_base_url = base
    config.nvidia_model = "local-fault-model"
    config.nvidia_timeout_ms = 250
    with TestClient(create_app(config)) as client:
        cid = client.post("/api/v1/collections", json={"name": "Provider"}).json()["id"]
        upload(client, cid)

        def query(generate=True):
            return client.post(
                "/api/v1/queries",
                json={
                    "collectionId": cid,
                    "question": "How long is Aster warranty coverage?",
                    "mode": "baseline",
                    "generate": generate,
                },
            )

        result = query()
        assert result.status_code == 200, result.text
        assert result.json()["answerStatus"] == "answered"
        assert state["requests"][-1][0] == "/v1/chat/completions"
        payload = state["requests"][-1][1]
        assert payload["model"] == "local-fault-model" and "[S1]" in payload["messages"][1]["content"]
        for status, code in (
            (401, "NVIDIA_UNAUTHORIZED"),
            (403, "NVIDIA_UNAUTHORIZED"),
            (429, "NVIDIA_RATE_LIMITED"),
            (500, "NVIDIA_UPSTREAM_ERROR"),
        ):
            state["status"] = status
            result = query()
            assert result.status_code == 502, result.text
            assert result.json()["error"]["code"] == code
            assert "local-dummy-secret" not in result.text
            assert query(False).status_code == 200
        state.update(status=200, delay=0.7)
        result = query()
        assert result.status_code == 504, result.text
        assert result.json()["error"]["code"] == "NVIDIA_TIMEOUT"
