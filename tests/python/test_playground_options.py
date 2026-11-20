import json

import numpy as np
import pytest
from conftest import upload
from test_provider import fake_nvidia as fake_nvidia

from server.embedding import LocalEmbedder
from server.playground_models import E5Embedder
from server.provider import SYSTEM

URL = "/api/v1/playground/run"


@pytest.mark.parametrize(
    "method,content,count",
    [
        ("rewrite", '{"queries":["current hardware warranty duration"]}', 1),
        ("decompose", '{"queries":["current warranty duration", "return window"]}', 2),
        (
            "hyde",
            '{"passage":"The current hardware warranty covers manufacturing defects for 24 months."}',
            1,
        ),
    ],
)
def test_query_preparation_uses_real_http_and_preserves_vectors(client, fake_nvidia, method, content, count):
    base, state = fake_nvidia
    state["content"] = content
    service = client.app.state.service
    service.config.nvidia_base_url = base
    service.config.nvidia_api_key = "local-test-key"
    request = {"queryPreparation": method, "untilStageId": "retrieve", "generate": False}
    result = client.post(URL, json=request)
    assert result.status_code == 200, result.text
    run = result.json()
    assert [stage["id"] for stage in run["stages"]] == [
        "load",
        "chunk",
        "embed",
        "query",
        "retrieve",
        "rerank",
        "generate",
    ]
    query, retrieval = run["stages"][3:5]
    assert query["status"] == retrieval["status"] == "complete"
    assert len(query["output"]["queries"]) == count
    assert query["output"]["execution"]["providerResponseReceived"]
    assert query["output"]["hypothetical"] == (method == "hyde")
    assert len(retrieval["output"]["searches"]) == count
    embed = run["stages"][2]["output"]
    for search in retrieval["output"]["searches"]:
        assert [p["coordinates"] for p in search["projection"]["points"]] == [
            p["coordinates"] for p in embed["projection"]["points"]
        ]
    assert len(state["requests"]) == 1
    assert state["requests"][0][1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert run["stages"][5]["status"] == "not_run"


def test_custom_sampling_prompt_and_citations_use_exact_http_payload(client, fake_nvidia):
    base, state = fake_nvidia
    service = client.app.state.service
    service.config.model_mode = "nvidia"
    service.config.nvidia_base_url = base
    service.config.nvidia_api_key = "local-test-key"
    request = {
        "prompt": "Answer briefly. Keep {literal braces}.",
        "temperature": 0.4,
        "topP": 0.85,
        "maxTokens": 256,
        "generationModel": "google/gemma-3-4b-it",
    }
    response = client.post(URL, json=request)
    assert response.status_code == 200, response.text
    generation = response.json()["stages"][-1]["output"]
    payload = state["requests"][-1][1]
    assert payload["temperature"] == 0.4 and payload["top_p"] == 0.85
    assert payload["max_tokens"] == 256 and payload["model"] == request["generationModel"]
    assert "top_k" not in payload
    assert payload["messages"][0]["content"] == request["prompt"] + "\n\n" + SYSTEM
    assert payload["messages"][0]["content"] == generation["systemPrompt"]
    assert payload["messages"][1]["content"] == generation["userPrompt"]
    assert generation["answerStatus"] == "answered"
    assert "local-test-key" not in json.dumps(response.json())


@pytest.mark.parametrize("content", ['{"queries":[]}', '{"queries":[1]}', '{"queries":["ok"]}', "not JSON"])
def test_bad_decomposition_is_rejected_without_generation(client, fake_nvidia, content):
    if content == '{"queries":["ok"]}':
        content = '{"queries":["ok","ok","ok","ok"]}'
    base, state = fake_nvidia
    service = client.app.state.service
    service.config.nvidia_base_url, service.config.nvidia_api_key = base, "local-test-key"
    state["content"] = content
    response = client.post(URL, json={"queryPreparation": "decompose"})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "QUERY_PREPARATION_INVALID"
    assert len(state["requests"]) == 1


@pytest.mark.parametrize(
    "settings",
    [
        {"temperature": -1},
        {"topP": 0},
        {"maxTokens": 10000},
        {"generationTopK": 20},
        {"ocr": True},
        {"embeddingModel": "untrusted/download"},
        {"untilStageId": "query"},
    ],
)
def test_new_settings_validate(client, settings):
    assert client.post(URL, json=settings).status_code == 400


def test_model_allowlist_and_options(client):
    options = client.get("/api/v1/playground/options").json()
    assert {m["id"] for m in options["embeddingModels"]} >= {"minilm-l6", "minilm-l3", "e5-small"}
    assert client.post(URL, json={"generationModel": "untrusted/model", "untilStage": 0}).status_code == 422
    assert "nvidia_api_key" not in json.dumps(options)


def test_query_provider_failure_keeps_completed_embedding_events(client, fake_nvidia):
    base, state = fake_nvidia
    state["status"] = 403
    service = client.app.state.service
    service.config.nvidia_base_url, service.config.nvidia_api_key = base, "local-test-key"
    response = client.post("/api/v1/playground/stream", json={"queryPreparation": "rewrite"})
    events = [json.loads(line) for line in response.text.splitlines()]
    completed = [event["stage"]["id"] for event in events if event["type"] == "stage_completed"]
    assert completed == ["load", "chunk", "embed"]
    query = [event for event in events if event["type"] == "stage_started"][-1]
    assert query["index"] == 3 and query["stage"]["id"] == "query"
    assert events[-1]["type"] == "run_error"
    assert events[-1]["error"]["code"] == "NVIDIA_UNAUTHORIZED"
    assert len(state["requests"]) == 1


def test_explicit_default_sampling_is_applied_and_http_rejection_is_not_model_reply(client, fake_nvidia):
    base, state = fake_nvidia
    state["status"] = 403
    service = client.app.state.service
    service.config.model_mode = "nvidia"
    service.config.nvidia_base_url, service.config.nvidia_api_key = base, "local-test-key"
    run = client.post(URL, json={"temperature": 0.2, "topP": 1.0, "maxTokens": 600}).json()
    output = run["stages"][-1]["output"]
    assert state["requests"][0][1]["top_p"] == 1.0
    assert output["error"]["code"] == "NVIDIA_UNAUTHORIZED"
    assert output["execution"]["providerRequestAttempted"] is True
    assert output["execution"]["providerResponseReceived"] is False
    assert output["execution"]["providerHttpStatus"] == 403


def test_hyde_never_enters_answer_evidence(client, fake_nvidia):
    base, state = fake_nvidia
    state["content"] = '{"passage":"Invented search passage: warranty covers accidental damage forever."}'
    service = client.app.state.service
    service.config.nvidia_base_url, service.config.nvidia_api_key = base, "local-test-key"
    run = client.post(URL, json={"queryPreparation": "hyde"}).json()
    generation = run["stages"][-1]["output"]
    assert run["sources"] and "Invented search passage" not in generation["userPrompt"]
    assert generation["userPrompt"].startswith("Question: " + run["configuration"]["question"])


def test_e5_uses_distinct_query_and_passage_prefixes():
    class Encoder:
        identity = "test"
        calls = []

        def embed_documents(self, texts):
            self.calls.append(texts)
            return [[1, 0]] * len(texts)

    encoder = Encoder()
    model = E5Embedder(encoder)
    model.embed_documents(["a passage"])
    model.embed_query("a question")
    assert encoder.calls == [["passage: a passage"], ["query: a question"]]


def test_bi_encoder_sort_uses_original_query_cosine(client):
    response = client.post(URL, json={"ranking": "bi-encoder", "embedding": "tfidf", "untilStage": 4})
    run = response.json()
    ranked = run["stages"][4]["output"]["after"]
    for candidate in ranked:
        assert candidate["scores"]["rerank"] == pytest.approx(
            np.dot(
                LocalEmbedder.stub(candidate["text"]), LocalEmbedder.stub(run["configuration"]["question"])
            ),
            abs=1e-6,
        )
    scores = [candidate["scores"]["rerank"] for candidate in ranked]
    assert scores == sorted(scores, reverse=True)


def test_cross_encoder_scores_sort_candidates_without_threshold(client, monkeypatch):
    playground = client.app.state.service.playground
    calls = []

    def scores(choice, question, texts):
        calls.append((choice, question, texts))
        return [-20 + index for index in range(len(texts))]

    monkeypatch.setattr(playground.models, "cross_scores", scores)
    run = client.post(URL, json={"ranking": "cross-encoder", "generate": False}).json()
    out = run["stages"][4]["output"]
    assert calls and out["after"][0]["id"] == out["before"][-1]["id"]
    assert run["sources"]


def test_ocr_reloads_original_file_without_changing_library(client, collection, monkeypatch):
    uploaded = upload(client, collection["id"])
    document = uploaded.json()["items"][0]
    service = client.app.state.service
    stored = service.store.get("document", document["id"])
    service.store.put("document", {**stored, "format": "pdf"})
    before = service.store.all("chunk", collection["id"])
    seen = []

    def parse(path, file_format, **kwargs):
        seen.append((path, file_format, kwargs))
        return {
            "parts": [{"text": "Recognized scanned warranty policy.", "location": {"page": 1}}],
            "ocrVerified": True,
        }

    monkeypatch.setattr(service, "parse", parse)
    run = client.post(
        URL,
        json={
            "source": "document",
            "documentId": document["id"],
            "collectionId": collection["id"],
            "ocr": True,
            "untilStage": 0,
        },
    ).json()
    assert run["stages"][0]["output"]["ocrVerified"]
    assert run["stages"][0]["output"]["documents"][0]["sections"][0]["text"].startswith("Recognized")
    assert seen == [(stored["filePath"], "pdf", {"ocr": True})]
    assert service.store.all("chunk", collection["id"]) == before


def test_real_scanned_pdf_ocr_isolated(tmp_path):
    pytest.importorskip("rapidocr")
    pytest.importorskip("onnxruntime")
    from PIL import Image, ImageDraw, ImageFont

    from server.parser import parse_isolated

    image = Image.new("RGB", (1400, 400), "white")
    font = (
        ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 44)
        if __import__("os").name == "nt"
        else ImageFont.load_default(size=44)
    )
    ImageDraw.Draw(image).text((60, 100), "Candidate name: Shweta Pardhi", fill="black", font=font)
    path = tmp_path / "scan.pdf"
    image.save(path, "PDF", resolution=150)
    parsed = parse_isolated(path, "pdf", ocr=True)
    assert parsed["ocrVerified"]
    assert "Shweta Pardhi" in " ".join(part["text"] for part in parsed["parts"])
