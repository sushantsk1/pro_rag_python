import math
import re
from time import perf_counter

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

STOP_WORDS = {"the", "and", "what", "how", "does", "for", "with", "from", "this", "that", "are"}


def tokenize(text: str) -> list[str]:
    return [
        word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 2 and word not in STOP_WORDS
    ]


def rerank(question: str, text: str, scores: dict) -> float:
    terms, words = set(tokenize(question)), set(tokenize(text))
    coverage = len(terms & words) / len(terms) if terms else 0
    phrase = float(question.lower().strip() in text.lower())
    semantic = max(0, scores.get("semantic") or 0)
    lexical = min(1, max(0, scores.get("lexical") or 0) / 5)
    return 0.45 * coverage + 0.30 * semantic + 0.20 * lexical + 0.05 * phrase


class RetrievalEngine:
    def __init__(self, store, embedder):
        self.store, self.embedder = store, embedder
        self.cache = {}
        self.chain = RunnableLambda(self._retrieve)

    def refresh(self):
        self.cache.clear()

    def retrieve(self, **request) -> dict:
        return self.chain.invoke(request)

    def _indexes(self, collection_id: str, document_ids: list[str], hybrid: bool):
        key = (collection_id, tuple(sorted(document_ids)), hybrid)
        if key in self.cache:
            return self.cache[key]
        ready = {d["id"] for d in self.store.all("document", collection_id) if d["status"] == "ready"}
        if document_ids:
            ready &= set(document_ids)
        chunks = [c for c in self.store.all("chunk", collection_id) if c["documentId"] in ready]
        docs = [Document(page_content=c["text"], metadata={"chunkId": c["id"]}, id=c["id"]) for c in chunks]
        lexical = (
            BM25Retriever.from_documents(docs, preprocess_func=tokenize)
            if any(tokenize(d.page_content) for d in docs)
            else None
        )
        vectors = None
        if hybrid and chunks:
            vectors = FAISS.from_embeddings(
                [(c["text"], c["embedding"]) for c in chunks],
                self.embedder,
                metadatas=[d.metadata for d in docs],
                ids=[c["id"] for c in chunks],
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )
        # A bounded cache avoids retaining an index for every possible filter combination.
        if len(self.cache) >= 8:
            self.cache.clear()
        self.cache[key] = (chunks, docs, lexical, vectors)
        return self.cache[key]

    def _retrieve(self, request: dict) -> dict:
        collection_id, question = request["collectionId"], request["question"]
        document_ids, top_k = request.get("documentIds", []), request.get("topK", 6)
        hybrid = request.get("mode", "hybrid") == "hybrid"
        started = perf_counter()
        chunks, docs, bm25, vectors = self._indexes(collection_id, document_ids, hybrid)
        stages, lexical_hits, semantic_hits = [], [], []
        limit = max(top_k * 4, 24)
        if bm25:
            scores = bm25.vectorizer.get_scores(tokenize(question))
            # BM25 may yield zero/negative scores for ubiquitous terms in a tiny corpus.
            # Keep only actual token matches, and expose a nonnegative comparable score.
            terms = set(tokenize(question))
            indexes = [i for i, doc in enumerate(docs) if terms.intersection(tokenize(doc.page_content))]
            indexes.sort(key=lambda i: float(scores[i]), reverse=True)
            lexical_hits = [(chunks[i]["id"], max(0.0, float(scores[i]))) for i in indexes[:limit]]
        stages.append(
            {
                "name": "lexical",
                "durationMs": round((perf_counter() - started) * 1000),
                "candidateCount": len(lexical_hits),
            }
        )
        if hybrid:
            started = perf_counter()
            if vectors:
                query_vector = self.embedder.embed_query(question)
                semantic_hits = [
                    (doc.metadata["chunkId"], float(score))
                    for doc, score in vectors.similarity_search_with_score_by_vector(query_vector, k=limit)
                    if math.isfinite(float(score)) and float(score) >= 0
                ]
            stages.append(
                {
                    "name": "semantic",
                    "durationMs": round((perf_counter() - started) * 1000),
                    "candidateCount": len(semantic_hits),
                }
            )
        started, ranked = perf_counter(), {}
        for kind, hits in (("lexical", lexical_hits), ("semantic", semantic_hits)):
            for index, (chunk_id, score) in enumerate(hits, 1):
                entry = ranked.setdefault(
                    chunk_id, {"lexical": None, "semantic": None, "fusion": 0.0, "rerank": None}
                )
                entry[kind] = score
                entry["fusion"] += 1 / (60 + index)
        fused = sorted(ranked, key=lambda key: ranked[key]["fusion"], reverse=True)[: max(top_k * 3, 18)]
        stages.append(
            {
                "name": "fusion",
                "durationMs": round((perf_counter() - started) * 1000),
                "candidateCount": len(fused),
            }
        )
        started = perf_counter()
        by_id = {c["id"]: c for c in chunks}
        for chunk_id in fused:
            ranked[chunk_id]["rerank"] = rerank(question, by_id[chunk_id]["text"], ranked[chunk_id])
        selected = sorted(fused, key=lambda key: ranked[key]["rerank"], reverse=True)[:top_k]
        stages.append(
            {
                "name": "rerank",
                "durationMs": round((perf_counter() - started) * 1000),
                "candidateCount": len(selected),
            }
        )
        documents = {d["id"]: d for d in self.store.all("document", collection_id)}
        sources = [
            {
                "chunkId": key,
                "documentId": by_id[key]["documentId"],
                "collectionId": collection_id,
                "documentName": documents[by_id[key]["documentId"]]["name"],
                "location": by_id[key]["location"],
                "snippet": by_id[key]["text"][:1000],
                "scores": ranked[key],
            }
            for key in selected
        ]
        return {
            "sources": sources,
            "confidence": ranked[selected[0]]["rerank"] if selected else 0,
            "trace": {
                "stages": stages,
                "reranker": "heuristic",
                "filters": {"documentIds": document_ids},
                "lexicalEngine": "bm25",
                "vectorEngine": "faiss",
                "framework": "langchain",
            },
        }
