import os

import pytest

from server.embedding import LocalEmbedder
from server.service import RagService


@pytest.mark.skipif(
    os.environ.get("RAG_TEST_REAL_EMBEDDINGS") != "1",
    reason="Opt in to downloading/running the real local model",
)
def test_real_minilm_embeddings_and_semantic_search(config):
    config.embedding_mode = "local"
    config.model_cache_dir = os.environ.get("MODEL_CACHE_DIR", "data/models")
    embedder = LocalEmbedder(config)
    vectors = embedder.embed_documents(
        ["The warranty covers manufacturing defects for two years.", "The restaurant serves pasta."]
    )
    assert all(len(vector) == 384 for vector in vectors)
    assert all(abs(sum(v * v for v in vector) - 1) < 1e-4 for vector in vectors)
    service = RagService(config, embedder=embedder)
    try:
        collection = service.create_collection({"name": "Real embeddings", "description": ""})
        service.upload(
            collection["id"],
            [
                {
                    "name": "coverage.txt",
                    "content": b"The warranty covers manufacturing defects for two years.",
                }
            ],
        )
        service.wait_for_jobs()
        result = service.query(
            {
                "collectionId": collection["id"],
                "question": "How long does warranty coverage last?",
                "mode": "hybrid",
                "generate": False,
                "topK": 6,
            }
        )
        assert result["sources"]
        assert result["sources"][0]["scores"]["semantic"] > 0
        assert service.config_status()["embeddingStatus"] == "ready"
    finally:
        service.close()
