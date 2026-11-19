import json
import socket
import threading
from time import sleep

import pytest
import requests
import uvicorn

from server.app import create_app
from server.errors import RagError
from server.playground import PlaygroundCancelled, PlaygroundRequest

URL = "/api/v1/playground/stream"


def events(client, settings=None):
    response = client.post(URL, json=settings or {"embedding": "tfidf", "generate": False})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in response.text.splitlines()]


def test_stream_reports_ordered_real_stages_and_timing(client):
    messages = events(client)
    assert messages[0]["type"] == "run_started"
    assert all(stage["status"] == "not_run" for stage in messages[0]["run"]["stages"])
    for index in range(6):
        started = next(
            i for i, e in enumerate(messages) if e["type"] == "stage_started" and e["index"] == index
        )
        ended = next(
            i for i, e in enumerate(messages) if e["type"] == "stage_completed" and e["index"] == index
        )
        assert started < ended
        assert messages[started]["stage"]["status"] == "running"
        assert messages[started]["stage"]["output"] is None
        assert messages[ended]["stage"]["durationMs"] >= 0
        if index < 5:
            assert ended < next(
                i for i, e in enumerate(messages) if e["type"] == "stage_started" and e["index"] == index + 1
            )
    assert messages[-1]["type"] == "run_completed"
    embedding = messages[-1]["run"]["stages"][2]["output"]
    assert embedding["projection"]["points"]
    assert embedding["embeddingDurationMs"] >= 0
    assert embedding["projectionDurationMs"] >= 0


def test_stream_partial_stops_before_embedding_and_generation(client, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("A chunk preview must stop before embedding or generation")

    monkeypatch.setattr(client.app.state.service.embedder, "embed_documents", unexpected)
    monkeypatch.setattr(client.app.state.service.generator, "generate", unexpected)
    messages = events(client, {"untilStage": 1})
    assert [e["index"] for e in messages if e["type"] == "stage_started"] == [0, 1]
    assert messages[-1]["run"]["stages"][2]["status"] == "not_run"


def test_generation_is_enabled_by_default(client, monkeypatch):
    calls = []

    def generate(question, sources, **kwargs):
        calls.append((question, sources))
        return {
            "answer": "A grounded answer [S1]",
            "citations": [{"marker": "S1", "chunkId": sources[0]["chunkId"]}],
            "answerStatus": "answered",
        }

    monkeypatch.setattr(client.app.state.service.generator, "generate", generate)
    for url in (URL, "/api/v1/playground/run"):
        response = client.post(url, json={"embedding": "tfidf"})
        run = json.loads(response.text.splitlines()[-1])["run"] if url == URL else response.json()
        assert run["configuration"]["generate"] is True
        assert run["stages"][5]["status"] == "complete"
        assert run["stages"][5]["output"]["answerStatus"] == "answered"
    assert len(calls) == 2


def test_embedding_failure_does_not_claim_success_or_advance(client, monkeypatch):
    def fail(*args, **kwargs):
        raise RagError(503, "EMBEDDING_UNAVAILABLE", "Local model unavailable")

    monkeypatch.setattr(client.app.state.service.embedder, "embed_documents", fail)
    messages = events(client, {"generate": False})
    assert [e["index"] for e in messages if e["type"] == "stage_completed"] == [0, 1]
    assert [e["index"] for e in messages if e["type"] == "stage_started"] == [0, 1, 2]
    assert messages[-1]["type"] == "run_error"
    assert messages[-1]["error"]["code"] == "EMBEDDING_UNAVAILABLE"
    assert not any(e["type"] == "run_completed" for e in messages)


def test_generation_failure_preserves_map_and_marks_only_generation_failed(client, monkeypatch):
    def fail(*args, **kwargs):
        raise RagError(502, "NVIDIA_UNAUTHORIZED", "Check model access")

    monkeypatch.setattr(client.app.state.service.generator, "generate", fail)
    messages = events(client, {"embedding": "tfidf"})
    run = messages[-1]["run"]
    assert run["stages"][2]["status"] == "complete"
    assert run["stages"][2]["output"]["projection"]["points"]
    assert run["stages"][5]["status"] == "error"


def test_disconnect_check_stops_next_stage_and_releases_gate(client):
    cancelled = threading.Event()
    playground = client.app.state.service.playground

    def emit(event):
        if event["type"] == "stage_completed" and event["index"] == 1:
            cancelled.set()

    with pytest.raises(PlaygroundCancelled):
        playground.run(PlaygroundRequest(), emit=emit, cancelled=cancelled)
    assert playground.gate.acquire(blocking=False)
    playground.gate.release()


def test_busy_stream_reports_retryable_failure(client):
    gate = client.app.state.service.playground.gate
    assert gate.acquire(blocking=False)
    try:
        messages = events(client)
        assert len(messages) == 1
        assert messages[0]["type"] == "run_error"
        assert messages[0]["error"]["status"] == 429
    finally:
        gate.release()


def test_live_stream_arrives_while_embedding_is_still_blocked(config, monkeypatch):
    app = create_app(config)
    entered, release = threading.Event(), threading.Event()
    embedder = app.state.service.embedder
    original = embedder.embed_documents

    def slow(texts):
        entered.set()
        if not release.wait(5):
            raise RuntimeError("Test did not release embedding")
        return original(texts)

    monkeypatch.setattr(embedder, "embed_documents", slow)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            sleep(0.02)
        assert server.started
        with requests.post(
            f"http://127.0.0.1:{port}{URL}", json={"generate": False}, stream=True, timeout=8
        ) as response:
            lines = response.iter_lines(chunk_size=1)
            received = []
            for line in lines:
                event = json.loads(line)
                received.append(event)
                if event["type"] == "stage_started" and event["index"] == 2:
                    break
            assert entered.wait(1)
            assert not release.is_set()
            assert [e["index"] for e in received if e["type"] == "stage_completed"] == [0, 1]
            assert not any(e["type"] == "stage_started" and e["index"] > 2 for e in received)
            release.set()
            rest = [json.loads(line) for line in lines]
            assert rest[-1]["type"] == "run_completed"
            assert rest[-1]["run"]["stages"][2]["output"]["projection"]["points"]
    finally:
        release.set()
        server.should_exit = True
        thread.join(timeout=5)
