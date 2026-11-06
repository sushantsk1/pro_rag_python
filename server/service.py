import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from .embedding import LocalEmbedder
from .errors import RagError
from .parser import SUPPORTED_FORMATS, parse_isolated
from .provider import AnswerGenerator, abstain
from .retrieval import RetrievalEngine
from .store import LocalStore, timestamp


def record_id():
    return str(uuid4())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class RagService:
    def __init__(self, config, *, embedder=None, parser=None, generator=None):
        self.config = config
        self.store = LocalStore(config.data_dir)
        self.embedder = embedder or LocalEmbedder(config)
        self.parse = parser or parse_isolated
        self.generator = generator or AnswerGenerator(config)
        self.engine = RetrievalEngine(self.store, self.embedder)
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-ingest")
        self.jobs = []
        self.ocr_status = "untested"
        for document in self.store.all("document"):
            if document["status"] in ("queued", "processing"):
                self.store.put(
                    "document",
                    {
                        **document,
                        "status": "failed",
                        "progress": 0,
                        "error": "Processing was interrupted. Retry this document.",
                        "updatedAt": timestamp(),
                    },
                )

    def collection_view(self, collection):
        docs = self.store.all("document", collection["id"])
        ready = [d for d in docs if d["status"] == "ready"]
        return {
            **collection,
            "documentCount": len(docs),
            "readyDocumentCount": len(ready),
            "chunkCount": sum(d.get("chunkCount", 0) for d in ready),
        }

    def get_collection(self, collection_id):
        collection = self.store.get("collection", collection_id)
        if not collection:
            raise RagError(404, "NOT_FOUND", "Collection not found.")
        return self.collection_view(collection)

    def list_collections(self, limit=20, cursor=None):
        return self.store.page([self.collection_view(c) for c in self.store.all("collection")], limit, cursor)

    def create_collection(self, body):
        with self.lock:
            if any(c["name"].lower() == body["name"].lower() for c in self.store.all("collection")):
                raise RagError(409, "DUPLICATE_COLLECTION", "A collection with this name already exists.")
            return self.collection_view(self.store.put("collection", {"id": record_id(), **body}))

    def update_collection(self, collection_id, patch):
        with self.lock:
            current = self.get_collection(collection_id)
            if "name" in patch and any(
                c["id"] != collection_id and c["name"].lower() == patch["name"].lower()
                for c in self.store.all("collection")
            ):
                raise RagError(409, "DUPLICATE_COLLECTION", "A collection with this name already exists.")
            return self.collection_view(
                self.store.put("collection", {**current, **patch, "updatedAt": timestamp()})
            )

    def delete_collection(self, collection_id):
        with self.lock:
            self.get_collection(collection_id)
            for document in self.store.all("document", collection_id):
                self.delete_document(document["id"])
            for kind in ("eval_case", "eval_run", "query"):
                self.store.delete_where(kind, lambda _: True, collection_id)
            self.store.delete("collection", collection_id)

    @staticmethod
    def public_document(document):
        return {key: value for key, value in document.items() if key not in ("filePath", "ocrRequested")}

    def get_document(self, document_id):
        document = self.store.get("document", document_id)
        if not document:
            raise RagError(404, "NOT_FOUND", "Document not found.")
        return self.public_document(document)

    def list_documents(self, collection_id, limit=20, cursor=None, status=None):
        self.get_collection(collection_id)
        return self.store.page(
            [
                self.public_document(d)
                for d in self.store.all("document", collection_id)
                if status is None or d["status"] == status
            ],
            limit,
            cursor,
        )

    def document_chunks(self, document_id, limit=20, cursor=None):
        document = self.get_document(document_id)
        if document["status"] != "ready":
            raise RagError(409, "DOCUMENT_NOT_READY", "This document is not searchable yet.")
        chunks = [
            {
                "id": c["id"],
                "chunkId": c["id"],
                "location": c["location"],
                "text": c["text"],
                "createdAt": c["createdAt"],
            }
            for c in self.store.all("chunk", document["collectionId"])
            if c["documentId"] == document_id
        ]
        return self.store.page(chunks, limit, cursor)

    def upload(self, collection_id, files, ocr=False):
        with self.lock:
            self.get_collection(collection_id)
            existing = self.store.all("document", collection_id)
            if not 1 <= len(files) <= 10:
                raise RagError(400, "VALIDATION_ERROR", "Upload 1 to 10 files.")
            if len(existing) + len(files) > self.config.document_limit:
                raise RagError(422, "COLLECTION_LIMIT", "This collection has reached its document limit.")
            hashes = {d["sha256"] for d in existing}
            validated = []
            for file in files:
                name = Path(file["name"].replace("\\", "/")).name[:180]
                file_format, content = Path(name).suffix[1:].lower(), file["content"]
                if file_format not in SUPPORTED_FORMATS:
                    raise RagError(
                        415, "UNSUPPORTED_FORMAT", f"Unsupported file format: {file_format or 'unknown'}."
                    )
                if not content:
                    raise RagError(400, "VALIDATION_ERROR", "Empty files cannot be uploaded.")
                if len(content) > self.config.upload_limit_bytes:
                    raise RagError(413, "FILE_TOO_LARGE", "File exceeds 25 MiB.")
                if file_format == "pdf" and not content.startswith(b"%PDF-"):
                    raise RagError(400, "VALIDATION_ERROR", f"{name} is not a PDF.")
                if file_format in ("docx", "pptx", "xlsx") and not content.startswith(b"PK\x03\x04"):
                    raise RagError(400, "VALIDATION_ERROR", f"{name} is not a valid Office archive.")
                digest = hashlib.sha256(content).hexdigest()
                if digest in hashes:
                    raise RagError(409, "DUPLICATE_DOCUMENT", f"{name} already exists in this collection.")
                hashes.add(digest)
                validated.append((file, name, file_format, digest))
            queued = []
            for file, name, file_format, digest in validated:
                document_id = record_id()
                file_path = self.store.files_dir / f"{document_id}.{file_format}"
                with file_path.open("xb") as handle:
                    handle.write(file["content"])
                document = self.store.put(
                    "document",
                    {
                        "id": document_id,
                        "collectionId": collection_id,
                        "name": name,
                        "format": file_format,
                        "mimeType": file.get("mimeType") or "application/octet-stream",
                        "sizeBytes": len(file["content"]),
                        "sha256": digest,
                        "status": "queued",
                        "progress": 0,
                        "extractionMode": None,
                        "locationSummary": "",
                        "chunkCount": 0,
                        "warning": None,
                        "error": None,
                        "filePath": str(file_path),
                        "ocrRequested": bool(ocr and file_format == "pdf"),
                    },
                )
                queued.append(self.public_document(document))
                self.enqueue(document_id)
            return {"items": queued}

    def enqueue(self, document_id):
        self.jobs = [future for future in self.jobs if not future.done()]
        self.jobs.append(self.executor.submit(self.process_document, document_id))

    def process_document(self, document_id):
        with self.lock:
            document = self.store.get("document", document_id)
            if not document or document["status"] != "queued":
                return
            self.store.put(
                "document", {**document, "status": "processing", "progress": 10, "updatedAt": timestamp()}
            )
        try:
            parsed = self.parse(
                document["filePath"],
                document["format"],
                ocr=document.get("ocrRequested", False),
                timeout=self.config.parse_timeout_seconds,
            )
            chunks, embedding_unavailable = [], False
            for index, part in enumerate(parsed["parts"]):
                text = str(part.get("text", "")).strip()
                if not text:
                    continue
                embedding = None
                if not embedding_unavailable:
                    try:
                        embedding = self.embedder.embed_query(text)
                    except RagError as cause:
                        if cause.code != "EMBEDDING_UNAVAILABLE":
                            raise
                        embedding_unavailable = True
                chunks.append(
                    {
                        "id": record_id(),
                        "documentId": document_id,
                        "collectionId": document["collectionId"],
                        "index": index,
                        "text": text,
                        "location": part["location"],
                        "embedding": embedding,
                        "embeddingModel": self.embedder.identity if embedding is not None else None,
                        "createdAt": timestamp(),
                    }
                )
                with self.lock:
                    current = self.store.get("document", document_id)
                    if not current or current["status"] != "processing":
                        return
                    self.store.put(
                        "document",
                        {
                            **current,
                            "progress": min(95, 20 + round(75 * (index + 1) / len(parsed["parts"]))),
                            "updatedAt": timestamp(),
                        },
                    )
            if not chunks:
                raise ValueError("No searchable text found.")
            warning = [parsed.get("warning")]
            if embedding_unavailable:
                warning.append(
                    "Local embeddings unavailable; baseline search remains available. Hybrid search will re-embed when the model is ready."
                )
            with self.lock:
                current = self.store.get("document", document_id)
                if not current or current["status"] != "processing":
                    return
                self.store.replace_chunks(
                    {
                        **current,
                        "extractionMode": parsed["mode"],
                        "locationSummary": ", ".join(sorted({c["location"]["type"] for c in chunks})),
                        "warning": " ".join(w for w in warning if w) or None,
                    },
                    chunks,
                )
                if document.get("ocrRequested"):
                    self.ocr_status = "ready" if parsed.get("ocrVerified") else "unavailable"
                self.engine.refresh()
        except Exception as cause:
            with self.lock:
                if document.get("ocrRequested"):
                    self.ocr_status = "unavailable"
                current = self.store.get("document", document_id)
                if current:
                    self.store.put(
                        "document",
                        {
                            **current,
                            "status": "failed",
                            "progress": 0,
                            "error": str(cause)[:220] or "Could not read document.",
                            "updatedAt": timestamp(),
                        },
                    )

    def retry_document(self, document_id):
        with self.lock:
            document = self.get_document(document_id)
            if document["status"] != "failed":
                raise RagError(409, "DOCUMENT_NOT_READY", "Only failed documents can be retried.")
            current = self.store.get("document", document_id)
            queued = self.store.put(
                "document",
                {**current, "status": "queued", "progress": 0, "error": None, "updatedAt": timestamp()},
            )
            self.enqueue(document_id)
            return self.public_document(queued)

    def delete_document(self, document_id):
        with self.lock:
            document = self.get_document(document_id)
            current = self.store.get("document", document_id)
            self.store.delete_where(
                "chunk", lambda c: c["documentId"] == document_id, document["collectionId"]
            )
            self.store.delete("document", document_id)
            # Original JS installations can store paths relative to the repository.
            file_path = Path(current["filePath"]).resolve()
            if file_path.is_relative_to(self.store.files_dir.resolve()):
                file_path.unlink(missing_ok=True)
            self.engine.refresh()

    def ensure_embeddings(self, collection_id):
        ready = {d["id"] for d in self.store.all("document", collection_id) if d["status"] == "ready"}
        missing = [
            c
            for c in self.store.all("chunk", collection_id)
            if c["documentId"] in ready
            and (
                not isinstance(c.get("embedding"), list) or c.get("embeddingModel") != self.embedder.identity
            )
        ]
        for start in range(0, len(missing), 32):
            batch = missing[start : start + 32]
            embeddings = self.embedder.embed_documents([c["text"] for c in batch])
            for chunk, embedding in zip(batch, embeddings, strict=True):
                self.store.put(
                    "chunk", {**chunk, "embedding": embedding, "embeddingModel": self.embedder.identity}
                )
        if missing:
            for doc in self.store.all("document", collection_id):
                if doc["id"] in ready and "Local embeddings unavailable" in (doc.get("warning") or ""):
                    warning = (
                        doc["warning"]
                        .replace(
                            "Local embeddings unavailable; baseline search remains available. Hybrid search will re-embed when the model is ready.",
                            "",
                        )
                        .strip()
                    )
                    self.store.put("document", {**doc, "warning": warning or None, "updatedAt": timestamp()})
            self.engine.refresh()

    def query(self, request):
        with self.lock:
            self.get_collection(request["collectionId"])
            for document_id in request.get("documentIds", []):
                doc = self.store.get("document", document_id)
                if not doc or doc["collectionId"] != request["collectionId"]:
                    raise RagError(404, "NOT_FOUND", "Filtered document not found in this collection.")
            started = perf_counter()
            if request["mode"] == "hybrid":
                self.ensure_embeddings(request["collectionId"])
            retrieved = self.engine.retrieve(**request)
            if request.get("generate", True) is False:
                generated = {"answer": None, "answerStatus": "retrieval_only", "citations": []}
            elif retrieved["confidence"] < 0.3:
                generated = abstain()
            else:
                generated = self.generator.generate(request["question"], retrieved["sources"])
            return self.store.put(
                "query",
                {
                    "id": record_id(),
                    "collectionId": request["collectionId"],
                    "question": request["question"],
                    "mode": request["mode"],
                    **generated,
                    "sources": retrieved["sources"],
                    "trace": retrieved["trace"],
                    "latencyMs": round((perf_counter() - started) * 1000),
                },
            )

    def list_queries(self, limit=20, cursor=None, collection_id=None):
        fields = {"id", "collectionId", "question", "answerStatus", "answer", "createdAt", "latencyMs"}
        return self.store.page(
            [{k: v for k, v in q.items() if k in fields} for q in self.store.all("query", collection_id)],
            limit,
            cursor,
        )

    def stats(self, collection_id=None):
        if collection_id:
            self.get_collection(collection_id)
        docs = self.store.all("document", collection_id)
        queries = sorted(self.store.all("query", collection_id), key=lambda q: q["createdAt"], reverse=True)
        runs = sorted(self.store.all("eval_run", collection_id), key=lambda r: r["createdAt"], reverse=True)
        return {
            "collectionCount": 1 if collection_id else len(self.store.all("collection")),
            "documentCount": len(docs),
            "readyDocumentCount": sum(d["status"] == "ready" for d in docs),
            "failedDocumentCount": sum(d["status"] == "failed" for d in docs),
            "processingDocumentCount": sum(d["status"] in ("queued", "processing") for d in docs),
            "chunkCount": sum(d.get("chunkCount", 0) for d in docs if d["status"] == "ready"),
            "formatCounts": [
                {"format": f, "count": sum(d["format"] == f for d in docs)}
                for f in sorted({d["format"] for d in docs})
            ],
            "recentQueries": [
                {k: q[k] for k in ("id", "question", "latencyMs", "answerStatus", "createdAt")}
                for q in queries[:5]
            ],
            "averageQueryLatencyMs": round(sum(q["latencyMs"] for q in queries) / len(queries))
            if queries
            else None,
            "evaluationRunCount": len(runs),
            "lastEvaluationRun": self.run_summary(runs[0]) if runs else None,
        }

    def create_case(self, body):
        self.get_collection(body["collectionId"])
        for doc_id in body["expectedDocumentIds"]:
            doc = self.store.get("document", doc_id)
            if not doc or doc["collectionId"] != body["collectionId"]:
                raise RagError(404, "NOT_FOUND", "Expected document not found in collection.")
        return self.store.put("eval_case", {"id": record_id(), **body})

    @staticmethod
    def run_summary(run):
        return {key: value for key, value in run.items() if key != "results"}

    def run_evaluation(self, request):
        with self.lock:
            collection_id, mode, top_k = request["collectionId"], request["mode"], request["topK"]
            self.get_collection(collection_id)
            cases = self.store.all("eval_case", collection_id)
            if not cases:
                raise RagError(
                    422, "NO_EVALUATION_CASES", "Add an evaluation case before running a benchmark."
                )
            if mode == "hybrid":
                self.ensure_embeddings(collection_id)
            normalized = sorted(
                ({k: c[k] for k in ("id", "question", "answerable", "expectedDocumentIds")} for c in cases),
                key=lambda c: c["id"],
            )
            for case in normalized:
                case["expectedDocumentIds"] = sorted(case["expectedDocumentIds"])
            ready = sorted(
                (
                    {"id": d["id"], "sha256": d["sha256"]}
                    for d in self.store.all("document", collection_id)
                    if d["status"] == "ready"
                ),
                key=lambda d: d["id"],
            )
            ready_ids = {d["id"] for d in ready}
            chunks = sorted(
                (
                    {k: c[k] for k in ("documentId", "index", "text", "location")}
                    for c in self.store.all("chunk", collection_id)
                    if c["documentId"] in ready_ids
                ),
                key=lambda c: (c["documentId"], c["index"]),
            )
            results = []
            for case in cases:
                started = perf_counter()
                try:
                    retrieved = self.engine.retrieve(
                        collectionId=collection_id,
                        question=case["question"],
                        documentIds=[],
                        mode=mode,
                        topK=top_k,
                    )
                    sources = retrieved["sources"]
                    first_rank = next(
                        (
                            i
                            for i, s in enumerate(sources, 1)
                            if s["documentId"] in case["expectedDocumentIds"]
                        ),
                        None,
                    )
                    generated = (
                        abstain()
                        if retrieved["confidence"] < 0.3
                        else self.generator.generate(case["question"], sources, fixture=True)
                    )
                    citation_valid = all(
                        any(s["chunkId"] == c["chunkId"] for s in sources) for c in generated["citations"]
                    )
                    citation_valid = citation_valid and bool(generated["citations"]) == case["answerable"]
                    results.append(
                        {
                            "caseId": case["id"],
                            "question": case["question"],
                            "expectedDocumentIds": case["expectedDocumentIds"],
                            "retrievedDocumentIds": list(dict.fromkeys(s["documentId"] for s in sources)),
                            "hit": first_rank is not None,
                            "reciprocalRank": 1 / first_rank if first_rank else 0,
                            "citationValid": citation_valid,
                            "abstentionCorrect": (generated["answerStatus"] != "abstained")
                            == case["answerable"],
                            "latencyMs": round((perf_counter() - started) * 1000),
                            "error": None,
                        }
                    )
                except Exception:
                    results.append(
                        {
                            "caseId": case["id"],
                            "question": case["question"],
                            "expectedDocumentIds": case["expectedDocumentIds"],
                            "retrievedDocumentIds": [],
                            "hit": False,
                            "reciprocalRank": 0,
                            "citationValid": False,
                            "abstentionCorrect": False,
                            "latencyMs": round((perf_counter() - started) * 1000),
                            "error": "Evaluation retrieval failed.",
                        }
                    )
            answerable = {c["id"] for c in cases if c["answerable"]}
            retrieval_results = [r for r in results if r["caseId"] in answerable]

            def average(field, rows=results):
                return sum(float(r[field]) for r in rows) / len(rows) if rows else None

            return self.store.put(
                "eval_run",
                {
                    "id": record_id(),
                    **request,
                    "caseCount": len(cases),
                    "caseSetFingerprint": fingerprint(normalized),
                    "corpusFingerprint": fingerprint(
                        {"embeddingModel": self.embedder.identity, "readyDocuments": ready, "chunks": chunks}
                    ),
                    "recallAtK": average("hit", retrieval_results),
                    "mrr": average("reciprocalRank", retrieval_results),
                    "citationValidity": average("citationValid"),
                    "abstentionAccuracy": average("abstentionCorrect"),
                    "averageLatencyMs": round(average("latencyMs")),
                    "results": results,
                },
            )

    def config_status(self):
        return {
            "nvidiaConfigured": bool(self.config.nvidia_api_key),
            "modelMode": self.config.model_mode,
            "model": self.config.nvidia_model,
            "embeddingStatus": self.embedder.status,
            "embeddingModel": self.config.embedding_model,
            "ocrAvailable": self.ocr_status == "ready",
            "ocrStatus": self.ocr_status,
            "supportedFormats": SUPPORTED_FORMATS,
            "uploadLimitBytes": self.config.upload_limit_bytes,
            "localOnly": True,
            "framework": "langchain",
            "vectorStore": "faiss",
            "lexicalSearch": "bm25",
        }

    def wait_for_jobs(self):
        for future in list(self.jobs):
            future.result(timeout=180)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.store.close()
