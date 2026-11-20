"""Experiment-only model choices; never mutate the library's embedding index."""

from collections import OrderedDict
from dataclasses import replace
from importlib.util import find_spec

import numpy as np

from .embedding import LocalEmbedder
from .errors import RagError

EMBEDDING_MODELS = {
    "minilm-l6": "sentence-transformers/all-MiniLM-L6-v2",
    "minilm-l3": "sentence-transformers/paraphrase-MiniLM-L3-v2",
    "e5-small": "intfloat/e5-small-v2",
}
CROSS_MODELS = {
    "minilm-l2": "cross-encoder/ms-marco-MiniLM-L-2-v2",
    "minilm-l6": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}
NVIDIA_MODELS = [
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-3.5-bolt-e4b",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
]


class E5Embedder:
    def __init__(self, embedder):
        self.embedder = embedder
        self.identity = embedder.identity + ":e5-prefixes"

    def embed_documents(self, texts):
        return self.embedder.embed_documents(["passage: " + text for text in texts])

    def embed_query(self, text):
        return self.embedder.embed_documents(["query: " + text])[0]


class ExperimentModels:
    def __init__(self, service):
        self.service = service
        self.embedders = OrderedDict()
        self.cross_encoders = OrderedDict()

    def embedder(self, choice):
        config = self.service.config
        name = EMBEDDING_MODELS.get(choice, config.embedding_model)
        if name == config.embedding_model and name != "intfloat/e5-small-v2":
            return self.service.embedder
        if name not in self.embedders:
            # Keep at most one additional encoder resident on CPU.
            self.embedders.clear()
            model = LocalEmbedder(replace(config, embedding_model=name))
            self.embedders[name] = E5Embedder(model) if name == "intfloat/e5-small-v2" else model
        return self.embedders[name]

    def cross_scores(self, choice, question, texts):
        if not texts:
            return []
        name = CROSS_MODELS[choice]
        try:
            if name not in self.cross_encoders:
                from sentence_transformers import CrossEncoder

                self.cross_encoders.clear()
                self.cross_encoders[name] = CrossEncoder(
                    name,
                    device="cpu",
                    cache_folder=str(self.service.config.model_cache_dir),
                    trust_remote_code=False,
                    max_length=512,
                )
            model = self.cross_encoders[name]
            # Use raw relevance logits; these are not answerability probabilities.
            import torch

            scores = np.asarray(
                model.predict(
                    [(question, text) for text in texts],
                    batch_size=16,
                    activation_fn=torch.nn.Identity(),
                    show_progress_bar=False,
                )
            ).reshape(-1)
            if len(scores) != len(texts) or not np.isfinite(scores).all():
                raise ValueError("Invalid cross-encoder scores")
            return scores.tolist()
        except Exception as cause:
            raise RagError(
                503,
                "CROSS_ENCODER_UNAVAILABLE",
                "Cross-encoder unavailable. Check the model download/cache and retry.",
            ) from cause

    def options(self):
        config = self.service.config
        return {
            "embeddingModels": [{"id": "configured", "name": config.embedding_model}]
            + [{"id": key, "name": name} for key, name in EMBEDDING_MODELS.items()],
            "crossEncoderModels": [{"id": key, "name": name} for key, name in CROSS_MODELS.items()],
            "generationModels": [
                {"id": name, "topKSupported": False}
                for name in dict.fromkeys([config.nvidia_model, *NVIDIA_MODELS])
            ],
            "defaultGenerationModel": config.nvidia_model,
            "ocrAvailable": all(
                find_spec(name) is not None for name in ("pypdfium2", "PIL", "rapidocr", "onnxruntime")
            ),
            "ocrEngine": "RapidOCR · local CPU · English",
            "samplingNote": "Token Top K is not exposed by these hosted model APIs; retrieval Top K remains available.",
        }
