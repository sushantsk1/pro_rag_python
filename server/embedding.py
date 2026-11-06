import hashlib
import math
import re
import threading

from langchain_core.embeddings import Embeddings

from .config import Config
from .errors import RagError


class LocalEmbedder(Embeddings):
    """Lazy local Sentence Transformers integration; fixtures require explicit stub mode."""

    def __init__(self, config: Config):
        self.config = config
        self.status = "ready" if config.embedding_mode == "stub" else "loading"
        self.model = None
        self.lock = threading.RLock()
        self.identity = (
            "stub:sha256:384:v1"
            if config.embedding_mode == "stub"
            else f"sentence-transformers:{config.embedding_model}:normalized:v1"
        )

    @staticmethod
    def stub(text: str) -> list[float]:
        vector = [0.0] * 384
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "little") % len(vector)
            vector[index] += 1
        norm = math.sqrt(sum(value * value for value in vector)) or 1
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.config.embedding_mode == "stub":
            return [self.stub(text) for text in texts]
        with self.lock:
            try:
                if self.model is None:
                    from langchain_huggingface import HuggingFaceEmbeddings

                    self.model = HuggingFaceEmbeddings(
                        model_name=self.config.embedding_model,
                        cache_folder=str(self.config.model_cache_dir),
                        model_kwargs={"device": "cpu", "trust_remote_code": False},
                        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
                    )
                result = self.model.embed_documents(texts)
                if len(result) != len(texts) or any(
                    not v or not all(math.isfinite(x) for x in v) for v in result
                ):
                    raise ValueError("Invalid embeddings")
                self.status = "ready"
                return result
            except Exception as cause:
                self.model = None
                self.status = "unavailable"
                raise RagError(
                    503,
                    "EMBEDDING_UNAVAILABLE",
                    "Local embedding model is unavailable. Check model download/cache and retry.",
                ) from cause

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
