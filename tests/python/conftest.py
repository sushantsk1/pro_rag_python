from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.app import create_app
from server.config import Config

FIXTURES = Path(__file__).resolve().parents[1] / "integration" / "fixtures"


@pytest.fixture
def config(tmp_path):
    return Config(
        data_dir=tmp_path / "data",
        model_cache_dir=tmp_path / "models",
        model_mode="stub",
        embedding_mode="stub",
    )


@pytest.fixture
def client(config):
    app = create_app(config)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def collection(client):
    response = client.post("/api/v1/collections", json={"name": "Test collection"})
    assert response.status_code == 201
    return response.json()


def upload(client, collection_id, name="policy.txt", content=None, ocr=False):
    content = (FIXTURES / name).read_bytes() if content is None else content
    response = client.post(
        "/api/v1/documents",
        data={"collectionId": collection_id, "ocr": str(ocr).lower()},
        files=[("files", (name, content, "application/octet-stream"))],
    )
    if response.status_code == 202:
        client.app.state.service.wait_for_jobs()
    return response
