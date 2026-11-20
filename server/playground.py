"""Bounded, read-only RAG experiments with inspectable intermediate results."""

import hashlib
import math
import re
import threading
from time import perf_counter, time
from typing import Literal
from uuid import uuid4

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rank_bm25 import BM25Okapi

from .errors import RagError
from .playground_models import NVIDIA_MODELS, ExperimentModels
from .playground_projection import project_embeddings
from .playground_provider import ExperimentProvider, system_prompt
from .provider import source_context
from .retrieval import rerank, tokenize
from .store import timestamp

MAX_CHARACTERS = 40000
MAX_CHUNKS = 256
SAMPLE_DOCUMENTS = [
    (
        "current-policy.md",
        "# Current device policy\nEffective September 1, 2026. Applies to purchases on or after this date.\n\n"
        "## Warranty\nThe current limited hardware warranty lasts 24 months from purchase. "
        "It covers manufacturing defects. Keep the receipt and device serial number. "
        "Accidental damage and water damage are excluded.\n\n"
        "## Returns\nAn unopened device may be returned within 30 calendar days of delivery. "
        "An opened device that is defective on arrival may be returned within 14 days. "
        "Original shipping charges are not refunded.",
    ),
    (
        "legacy-policy.md",
        "# Legacy device policy\nApplies to purchases before September 1, 2026.\n\n"
        "## Warranty\nThe legacy limited hardware warranty lasts 12 months from purchase. "
        "It covers manufacturing defects and excludes accidental damage. "
        "This historical policy does not apply to purchases made on or after September 1, 2026.\n\n"
        "## Returns\nThe legacy return window for unopened devices is 21 days from delivery.",
    ),
    (
        "support-guide.md",
        "# Customer support\n\n## Opening hours\nLive support is available Monday through Friday, "
        "09:00 to 18:00 India Standard Time. The support portal accepts tickets at any hour.\n\n"
        "## Repairs\nAn authorized service centre assesses hardware claims. "
        "Back up personal files before sending a device for repair. "
        "Support can explain the policy but cannot extend its duration.",
    ),
]


class PlaygroundRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["sample", "text", "document"] = "sample"
    text: str = Field(default="", max_length=MAX_CHARACTERS)
    documentId: str | None = Field(default=None, min_length=36, max_length=36)
    collectionId: str | None = Field(default=None, min_length=36, max_length=36)
    question: str = Field(default="What is the current hardware warranty?", min_length=2, max_length=2000)
    loader: Literal["sections", "plain"] = "sections"
    chunking: Literal["fixed", "recursive", "parent-child", "semantic"] = "recursive"
    chunkSize: int = Field(default=500, ge=120, le=1500, strict=True)
    overlap: int = Field(default=80, ge=0, le=200, strict=True)
    semanticThreshold: float = Field(default=0.55, ge=0, le=1, allow_inf_nan=False)
    embedding: Literal["local", "tfidf"] = "local"
    embeddingModel: Literal["configured", "minilm-l6", "minilm-l3", "e5-small"] = "configured"
    ocr: bool = Field(default=False, strict=True)
    queryPreparation: Literal["none", "rewrite", "decompose", "hyde"] = "none"
    retrieval: Literal["keyword", "semantic", "hybrid"] = "hybrid"
    ranking: Literal["none", "heuristic", "bi-encoder", "cross-encoder"] = "heuristic"
    crossEncoderModel: Literal["minilm-l2", "minilm-l6"] = "minilm-l2"
    generationModel: str | None = Field(default=None, max_length=160)
    prompt: str = Field(default="", max_length=6000)
    temperature: float = Field(default=0.2, ge=0, le=1, allow_inf_nan=False)
    topP: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    generationTopK: int | None = Field(default=None, ge=1, le=100, strict=True)
    maxTokens: int = Field(default=600, ge=64, le=2048, strict=True)
    topK: int = Field(default=6, ge=1, le=10, strict=True)
    generate: bool = Field(default=True, strict=True)
    untilStage: int = Field(default=5, ge=0, le=5, strict=True)
    untilStageId: Literal["load", "chunk", "embed", "query", "retrieve", "rerank", "generate"] | None = None

    @field_validator("question")
    @classmethod
    def clean_question(cls, value):
        if len(value.strip()) < 2:
            raise ValueError("Enter a question.")
        return value.strip()

    @model_validator(mode="after")
    def check_settings(self):
        if self.generationTopK is not None:
            raise ValueError("Token Top K is not exposed by the supported hosted endpoints. Use Top P instead.")
        if self.untilStageId == "query" and self.queryPreparation == "none":
            raise ValueError("Enable query preparation before running that stage.")
        if self.ocr and self.source != "document":
            raise ValueError("OCR requires a library PDF document.")
        if self.overlap >= self.chunkSize:
            raise ValueError("Overlap must be smaller than the chunk size.")
        if self.chunking == "semantic" and self.overlap:
            raise ValueError("Semantic grouping uses sentence boundaries; set overlap to zero.")
        if self.source == "text" and len(self.text.strip()) < 2:
            raise ValueError("Enter source text.")
        if self.source == "document" and (not self.documentId or not self.collectionId):
            raise ValueError("Choose a document and its collection.")
        return self


def split_text(text, size, overlap, method="recursive"):
    if method == "fixed":
        return [(start, text[start : start + size]) for start in range(0, len(text), size - overlap)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, add_start_index=True)
    return [(doc.metadata["start_index"], doc.page_content) for doc in splitter.create_documents([text])]


def tfidf_vectors(texts, question):
    """Normalized TF-IDF with corpus-only IDF; no download or ML model needed."""
    tokenized = [tokenize(text) for text in texts]
    vocabulary = sorted({word for words in tokenized for word in words})
    if not vocabulary:
        raise RagError(422, "NO_SEARCHABLE_TOKENS", "The input contains no searchable words.")
    index = {word: i for i, word in enumerate(vocabulary)}
    idf = np.array(
        [
            math.log((1 + len(texts)) / (1 + sum(word in words for words in tokenized))) + 1
            for word in vocabulary
        ]
    )

    def vector(words):
        row = np.zeros(len(vocabulary), dtype=np.float32)
        for word in words:
            if word in index:
                row[index[word]] += 1
        row *= idf
        norm = np.linalg.norm(row)
        return row / norm if norm else row

    return np.array([vector(words) for words in tokenized]), vector(tokenize(question)), vocabulary


class PlaygroundCancelled(Exception):
    """A disconnected experiment must not advance to another stage."""


class Playground:
    def __init__(self, service):
        self.service = service
        self.models = ExperimentModels(service)
        self.provider = ExperimentProvider(service.config)
        self.gate = threading.BoundedSemaphore(1)

    def load(self, request):
        if request.source == "sample":
            documents = [
                {"id": f"D{i}", "name": name, "sections": [{"text": text, "location": {}}]}
                for i, (name, text) in enumerate(SAMPLE_DOCUMENTS, 1)
            ]
            note = "Built-in fictional documents. No library records are changed."
        elif request.source == "text":
            documents = [
                {"id": "D1", "name": "pasted-text.txt", "sections": [{"text": request.text, "location": {}}]}
            ]
            note = "Pasted text is used only for this experiment."
        else:
            with self.service.lock:
                doc = self.service.get_document(request.documentId)
                if doc["collectionId"] != request.collectionId:
                    raise RagError(404, "NOT_FOUND", "Document not found in this collection.")
                if doc["status"] != "ready" and not (request.ocr and doc["status"] == "failed"):
                    raise RagError(409, "DOCUMENT_NOT_READY", "Wait for document processing to finish.")
                chunks = sorted(
                    (
                        c
                        for c in self.service.store.all("chunk", request.collectionId)
                        if c["documentId"] == doc["id"]
                    ),
                    key=lambda c: c["index"],
                )
                stored = self.service.store.get("document", doc["id"])
            if request.ocr:
                if stored["format"] != "pdf":
                    raise RagError(422, "OCR_REQUIRES_PDF", "Choose a PDF document to use OCR.")
                parsed = self.service.parse(stored["filePath"], "pdf", ocr=True)
                chunks = parsed["parts"]
            documents = [
                {
                    "id": doc["id"],
                    "name": doc["name"],
                    "sections": [{"text": c["text"], "location": c["location"]} for c in chunks],
                }
            ]
            note = (
                "Uses existing extracted library sections, including their original overlap. "
                "It does not reparse or replace the uploaded file."
            )
            if request.ocr:
                note = ("Re-extracted the original PDF with OCR on pages without text. "
                        "Library chunks and indexes stay unchanged. " + " ".join(parsed.get("warnings", [])))
        source_fingerprint = hashlib.sha256(
            repr([(d["id"], [s["text"] for s in d["sections"]]) for d in documents]).encode()
        ).hexdigest()
        remaining = MAX_CHARACTERS
        original_size = sum(len(s["text"]) for d in documents for s in d["sections"])
        loaded = []
        for doc in documents:
            sections = []
            for section in doc["sections"]:
                value = section["text"][:remaining]
                remaining -= len(value)
                if value.strip():
                    sections.append({**section, "text": value})
            if not sections:
                continue
            if request.loader == "plain":
                sections = [{"text": "\n\n".join(s["text"] for s in sections), "location": {}}]
            elif request.source != "document":
                structured = []
                for section in sections:
                    for value in re.split(r"\n(?=## )", section["text"]):
                        heading = re.search(r"^#{1,2}\s+(.+)", value, re.M)
                        structured.append(
                            {
                                "text": value,
                                "location": {"type": "heading", "heading": heading[1]} if heading else {},
                            }
                        )
                sections = structured
            loaded.append({**doc, "sections": sections})
        if not loaded:
            raise RagError(422, "NO_TEXT", "No extracted text is available for this experiment.")
        return loaded, {
            "ocrRequested": request.ocr,
            "ocrVerified": bool(request.ocr and parsed.get("ocrVerified")),
            "note": note,
            "fingerprint": source_fingerprint,
            "truncated": original_size > MAX_CHARACTERS,
            "availableCharacters": original_size,
            "characterLimit": MAX_CHARACTERS,
        }

    def chunk(self, documents, request):
        chunks, parents, boundaries = [], [], []
        for doc in documents:
            for section_index, section in enumerate(doc["sections"]):
                value, location = section["text"], section["location"]
                blocks = [(0, value, None)]
                if request.chunking == "parent-child":
                    blocks = []
                    for offset, parent_text in split_text(value, min(4000, request.chunkSize * 4), 0):
                        parent_id = f"P{len(parents) + 1:03}"
                        parents.append(
                            {
                                "id": parent_id,
                                "documentId": doc["id"],
                                "text": parent_text,
                                "location": location,
                                "section": section_index,
                                "start": offset,
                            }
                        )
                        blocks.append((offset, parent_text, parent_id))
                for block_offset, block, parent_id in blocks:
                    method = "fixed" if request.chunking == "fixed" else "recursive"
                    if request.chunking == "semantic":
                        spans = list(re.finditer(r"\S[\s\S]*?(?:[.!?](?=\s|$)|(?=\n{2,})|$)", block))
                        spans = [span for span in spans if span[0].strip()]
                        if len(spans) > 400:
                            raise RagError(
                                422, "PLAYGROUND_LIMIT", "Use fewer than 400 sentences for semantic grouping."
                            )
                        vectors = (
                            self.models.embedder(request.embeddingModel).embed_documents([span[0] for span in spans])
                            if spans
                            else []
                        )
                        groups, start, end = [], 0, 0
                        for i, span in enumerate(spans):
                            similarity = float(np.dot(vectors[i - 1], vectors[i])) if i else 1
                            if i:
                                boundaries.append(
                                    {
                                        "documentId": doc["id"],
                                        "offset": span.start(),
                                        "similarity": similarity,
                                        "split": similarity < request.semanticThreshold
                                        or span.end() - start > request.chunkSize,
                                    }
                                )
                            if i and (
                                similarity < request.semanticThreshold
                                or span.end() - start > request.chunkSize
                            ):
                                groups.extend(
                                    (start + off, text)
                                    for off, text in split_text(block[start:end], request.chunkSize, 0)
                                )
                                start = span.start()
                            if not i:
                                start = span.start()
                            end = span.end()
                        if spans:
                            groups.extend(
                                (start + off, text)
                                for off, text in split_text(block[start:end], request.chunkSize, 0)
                            )
                        pieces = groups
                    else:
                        pieces = split_text(block, request.chunkSize, request.overlap, method)
                    for offset, text in pieces:
                        if not text.strip():
                            continue
                        chunks.append(
                            {
                                "id": f"C{len(chunks) + 1:03}",
                                "documentId": doc["id"],
                                "documentName": doc["name"],
                                "text": text,
                                "location": location,
                                "section": section_index,
                                "start": block_offset + offset,
                                "end": block_offset + offset + len(text),
                                "parentId": parent_id,
                            }
                        )
                        if len(chunks) > MAX_CHUNKS:
                            raise RagError(
                                422,
                                "PLAYGROUND_LIMIT",
                                "This creates more than 256 chunks. Increase the chunk size or reduce overlap.",
                            )
        return chunks, parents, boundaries

    def run(self, request, emit=None, cancelled=None):
        if not self.gate.acquire(blocking=False):
            raise RagError(429, "PLAYGROUND_BUSY", "Another experiment is running. Try again shortly.")
        try:
            return self._run(request, emit, cancelled)
        finally:
            self.gate.release()

    def _run(self, request, emit=None, cancelled=None):
        if request.generationModel and request.generationModel not in {*NVIDIA_MODELS, self.service.config.nvidia_model}:
            raise RagError(422, "MODEL_NOT_ALLOWED", "Choose a model from the experiment model list.")
        started = perf_counter()
        stage_names = ["load", "chunk", "embed", "retrieve", "rerank", "generate"]
        if request.queryPreparation != "none":
            stage_names.insert(3, "query")
        stages = [
            {"id": name, "status": "not_run", "durationMs": None, "input": None, "output": None}
            for name in stage_names
        ]
        chunks, sources, vectors, parents = [], [], [], []
        config = request.model_dump()
        # Source text is present in the load stage; avoid duplicating it in settings.
        config.pop("text")
        result = {
            "id": str(uuid4()),
            "createdAt": timestamp(),
            "configuration": config,
            "stages": stages,
            "sources": sources,
        }
        if emit:
            emit({"type": "run_started", "run": result})

        def start_stage(index, message):
            index = actual_index(index)
            if cancelled is not None and cancelled.is_set():
                raise PlaygroundCancelled()
            stages[index].update(status="running", startedAtMs=round(time() * 1000), message=message)
            if emit:
                emit({"type": "stage_started", "index": index, "stage": stages[index]})
            return perf_counter()

        def progress(index, message):
            index = actual_index(index)
            stages[index]["message"] = message
            if emit:
                emit({"type": "stage_progress", "index": index, "message": message})

        def finish():
            result["summary"] = {
                "documentCount": len(documents),
                "chunkCount": len(chunks),
                "vectorCount": len(vectors),
                "sourceCount": len(sources),
                "dimensions": len(vectors[0]) if len(vectors) else 0,
                "durationMs": round((perf_counter() - started) * 1000, 2),
            }
            return result

        def stage(index, clock, input_value, output_value, status="complete"):
            index = actual_index(index)
            stages[index].update(
                status=status,
                durationMs=round((perf_counter() - clock) * 1000, 2),
                input=input_value,
                output=output_value,
            )
            if emit:
                emit({"type": "stage_completed", "index": index, "stage": stages[index]})

        def actual_index(index):
            name = index if isinstance(index, str) else ("load", "chunk", "embed", "retrieve", "rerank", "generate")[index]
            return stage_names.index(name)

        def stopped(index):
            target = request.untilStageId or ("load", "chunk", "embed", "retrieve", "rerank", "generate")[request.untilStage]
            return stage_names[actual_index(index)] == target

        clock = start_stage(0, "Loading source text")
        documents, source_info = self.load(request)
        result["sourceFingerprint"] = source_info["fingerprint"]
        stage(
            0,
            clock,
            {"source": request.source, "documentId": request.documentId},
            {"documents": documents, **source_info},
        )
        if stopped(0):
            return finish()

        clock = start_stage(
            1,
            "Grouping sentences with local embeddings"
            if request.chunking == "semantic"
            else "Splitting source into chunks",
        )
        chunks, parents, boundaries = self.chunk(documents, request)
        if not chunks:
            raise RagError(422, "NO_CHUNKS", "No searchable chunks were created.")
        stage(
            1,
            clock,
            {
                "documentIds": [d["id"] for d in documents],
                "strategy": request.chunking,
                "chunkSize": request.chunkSize,
                "overlap": request.overlap,
            },
            {
                "chunks": chunks,
                "parents": parents,
                "boundaries": boundaries,
                "semanticBoundaryModel": self.models.embedder(request.embeddingModel).identity
                if request.chunking == "semantic"
                else None,
            },
        )
        if stopped(1):
            return finish()

        clock = start_stage(
            2,
            "Computing TF-IDF vectors"
            if request.embedding == "tfidf"
            else "Creating chunk and question embeddings; the local model may need to load",
        )
        if request.embedding == "tfidf":
            vectors, query_vector, vocabulary = tfidf_vectors([c["text"] for c in chunks], request.question)
            model = "TF-IDF (corpus vocabulary)"
            embedding_mode = "tfidf"
        else:
            vectors = np.asarray(
                self.models.embedder(request.embeddingModel).embed_documents([c["text"] for c in chunks]), dtype=np.float32
            )
            query_vector = np.asarray(self.models.embedder(request.embeddingModel).embed_query(request.question), dtype=np.float32)
            vocabulary = None
            model = self.models.embedder(request.embeddingModel).identity
            embedding_mode = self.service.config.embedding_mode
        embedding_duration = round((perf_counter() - clock) * 1000, 2)
        progress(2, "Building the vector index and embedding map")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        projection_started = perf_counter()
        projection = project_embeddings(chunks, vectors, query_vector)
        projection_duration = round((perf_counter() - projection_started) * 1000, 2)
        stage(
            2,
            clock,
            {"chunkIds": [c["id"] for c in chunks], "question": request.question},
            {
                "model": model,
                "mode": embedding_mode,
                "dimensions": vectors.shape[1],
                "index": "FAISS IndexFlatIP",
                "vocabulary": vocabulary,
                "vectors": [
                    {"chunkId": c["id"], "values": row.tolist(), "norm": float(np.linalg.norm(row))}
                    for c, row in zip(chunks, vectors, strict=True)
                ],
                "queryVector": query_vector.tolist(),
                "projection": projection,
                "embeddingDurationMs": embedding_duration,
                "projectionDurationMs": projection_duration,
            },
        )
        if stopped(2):
            return finish()

        queries = [request.question]
        search_vectors = [query_vector]
        search_maps = []
        if request.queryPreparation != "none":
            clock = start_stage("query", "Preparing search text with NVIDIA")
            prepared = self.provider.prepare(request)
            queries = prepared["queries"]
            search_vectors = []
            for query in queries:
                if request.embedding == "tfidf":
                    vector = tfidf_vectors([c["text"] for c in chunks], query)[1]
                elif request.queryPreparation == "hyde":
                    vector = np.asarray(self.models.embedder(request.embeddingModel).embed_documents([query])[0], dtype=np.float32)
                else:
                    vector = np.asarray(self.models.embedder(request.embeddingModel).embed_query(query), dtype=np.float32)
                search_vectors.append(vector)
                search_maps.append({"text": query, "projection": project_embeddings(chunks, vectors, vector)})
            prepared["searches"] = search_maps
            stage("query", clock, {"question": request.question}, prepared)
            if stopped("query"):
                return finish()

        clock = start_stage(3, "Searching for relevant chunks")
        limit = max(request.topK * 4, 24)
        all_hits = []
        words = [tokenize(c["text"]) for c in chunks]
        bm25 = BM25Okapi(words) if any(words) else None
        for query, search_vector in zip(queries, search_vectors, strict=True):
            lexical, semantic = [], []
            lexical_query = request.question if request.queryPreparation == "hyde" else query
            if request.retrieval in ("keyword", "hybrid") and bm25:
                scores = bm25.get_scores(tokenize(lexical_query))
                terms = set(tokenize(lexical_query))
                lexical = sorted(
                    [
                        (i, max(0.0, float(scores[i])))
                        for i, text in enumerate(words)
                        if terms.intersection(text)
                    ],
                    key=lambda pair: pair[1],
                    reverse=True,
                )[:limit]
            if request.retrieval in ("semantic", "hybrid") and np.linalg.norm(search_vector):
                distances, positions = index.search(
                    np.asarray([search_vector], dtype=np.float32), min(limit, len(chunks))
                )
                semantic = [
                    (int(i), float(score))
                    for i, score in zip(positions[0], distances[0], strict=True)
                    if i >= 0 and math.isfinite(float(score)) and score > 0
                ]
            all_hits.extend([("lexical", lexical), ("semantic", semantic)])
        # Aggregate best raw scores for display; RRF uses every per-query ranking.
        def best_hits(kind):
            best = {}
            for channel, hits in all_hits:
                if channel == kind:
                    for position, score in hits:
                        best[position] = max(best.get(position, -math.inf), score)
            return sorted(best.items(), key=lambda pair: pair[1], reverse=True)

        lexical, semantic = best_hits("lexical"), best_hits("semantic")
        fused = {}
        for kind, hits in all_hits:
            for rank, (i, score) in enumerate(hits, 1):
                item = fused.setdefault(
                    i,
                    {
                        **chunks[i],
                        "scores": {"lexical": None, "semantic": None, "fusion": 0.0, "rerank": None},
                    },
                )
                item["scores"][kind] = max(item["scores"][kind] or 0, score)
                item["scores"]["fusion"] += 1 / (60 + rank)
        candidates = sorted(fused.values(), key=lambda c: c["scores"]["fusion"], reverse=True)[
            : max(3 * request.topK, 18)
        ]
        stage(
            3,
            clock,
            {"question": request.question, "method": request.retrieval, "topK": request.topK},
            {
                "searches": search_maps,
                "queries": queries,
                "candidates": candidates,
                "lexicalHits": [{"chunkId": chunks[i]["id"], "score": s} for i, s in lexical],
                "semanticHits": [{"chunkId": chunks[i]["id"], "score": s} for i, s in semantic],
                "fusion": "Reciprocal rank fusion (k=60)",
            },
        )
        if stopped(3):
            return finish()

        clock = start_stage(4, "Ranking candidates and selecting context")
        ranked = [
            {
                **candidate,
                "scores": {
                    **candidate["scores"],
                    "rerank": rerank(request.question, candidate["text"], candidate["scores"]),
                },
            }
            for candidate in candidates
        ]
        if request.ranking == "cross-encoder":
            scores = self.models.cross_scores(request.crossEncoderModel, request.question, [c["text"] for c in ranked])
            for candidate, score in zip(ranked, scores, strict=True):
                candidate["scores"]["rerank"] = score
        elif request.ranking == "bi-encoder":
            if request.embedding == "tfidf" and ranked:
                encoder = self.models.embedder(request.embeddingModel)
                rank_vectors = np.asarray(encoder.embed_documents([c["text"] for c in ranked]), dtype=np.float32)
                rank_query = np.asarray(encoder.embed_query(request.question), dtype=np.float32)
            else:
                by_id = {chunk["id"]: i for i, chunk in enumerate(chunks)}
                rank_vectors = [vectors[by_id[c["id"]]] for c in ranked]
                rank_query = query_vector
            for candidate, vector in zip(ranked, rank_vectors, strict=True):
                candidate["scores"]["rerank"] = float(vector @ rank_query)
        if request.ranking != "none":
            ranked.sort(key=lambda c: c["scores"]["rerank"], reverse=True)
        selected = ranked[: request.topK]
        by_parent = {p["id"]: p for p in parents}
        used_parents = set()
        for child in selected:
            parent = by_parent.get(child["parentId"])
            if parent and parent["id"] in used_parents:
                continue
            if parent:
                used_parents.add(parent["id"])
            sources.append(
                {
                    "chunkId": child["id"],
                    "documentId": child["documentId"],
                    "documentName": child["documentName"],
                    "location": child["location"],
                    "snippet": parent["text"] if parent else child["text"],
                    "parentId": child["parentId"],
                    "scores": child["scores"],
                }
            )
        stage(
            4,
            clock,
            {"candidateIds": [c["id"] for c in candidates], "method": request.ranking},
            {
                "before": candidates,
                "after": ranked,
                "selected": selected,
                "sources": sources,
                "note": "Parent context replaces retrieved children; duplicate parents are sent once."
                if parents
                else "Selected chunk text becomes answer context.",
            },
        )
        if stopped(4):
            return finish()

        clock = start_stage(
            5,
            "Generating a grounded answer with NVIDIA"
            if request.generate
            else "Preparing the evidence prompt",
        )
        context = source_context(sources)
        generation = {
            "systemPrompt": system_prompt(request),
            "userPrompt": f"Question: {request.question}\n\nEvidence:\n{context}",
            "answer": None,
            "citations": [],
            "answerStatus": "disabled",
            "providerMode": self.service.config.model_mode,
            "retrievalScore": selected[0]["scores"]["rerank"] if selected else None,
            "execution": {
                "generatorInvoked": False,
                "providerRequestAttempted": False,
                "providerResponseReceived": False,
                "decision": "disabled",
            },
        }
        status = "skipped"
        if request.generate:
            try:
                # Ranking scores are not calibrated answerability probabilities.
                # Send retrieved context to the grounded generator even for entity
                # questions with little literal overlap ("candidate name", etc.).
                execution = generation["execution"]
                execution.update(generatorInvoked=True, decision="not_started")
                # Explicit controls must be sent even when equal to UI defaults;
                # an SDK's own sampling defaults can otherwise differ.
                customized = bool({"generationModel", "prompt", "temperature", "topP", "maxTokens"}
                                  & request.model_fields_set)
                if customized and self.service.config.model_mode != "stub":
                    generated = self.provider.generate(request, sources, execution)
                else:
                    generated = self.service.generator.generate(request.question, sources, trace=execution)
                generation.update(generated)
                status = "complete"
            except RagError as error:
                if generation["execution"]["decision"] == "not_started":
                    generation["execution"]["decision"] = "provider_error"
                generation["answerStatus"] = "error"
                generation["error"] = {"code": error.code, "message": error.message}
                status = "error"
        stage(
            5,
            clock,
            {"question": request.question, "sourceIds": [s["chunkId"] for s in sources]},
            generation,
            status,
        )
        return finish()
