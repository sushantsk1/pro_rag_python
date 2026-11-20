import json
import re
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from .config import load_config
from .errors import RagError
from .playground import PlaygroundRequest
from .playground_stream import stream_experiment
from .service import RagService
from .store import timestamp

PREFIX = "/api/v1"
ORIGINS = {"http://127.0.0.1:5173", "http://localhost:5173"}


def problem(error: RagError):
    return JSONResponse(
        status_code=error.status,
        content={
            "error": {
                "code": error.code,
                "message": error.message,
                "details": error.details,
                "requestId": str(uuid4()),
            }
        },
    )


def string(value, field, minimum, maximum):
    if not isinstance(value, str) or not minimum <= len(value.strip()) <= maximum:
        raise RagError(
            400, "VALIDATION_ERROR", f"Invalid {field}.", {field: f"Use {minimum} to {maximum} characters."}
        )
    return value.strip()


def boolean(value, field, default=True):
    if value is None:
        return default
    if type(value) is not bool:
        raise RagError(400, "VALIDATION_ERROR", f"Invalid {field}.", {field: "Use true or false."})
    return value


def count(value, field, default=6, minimum=1, maximum=20):
    if value is None:
        return default
    try:
        number = int(value)
        if isinstance(value, bool) or str(number) != str(value) or not minimum <= number <= maximum:
            raise ValueError
        return number
    except (ValueError, TypeError):
        raise RagError(400, "VALIDATION_ERROR", f"Invalid {field}.") from None


def ids(value, field):
    if value is None:
        return []
    if (
        not isinstance(value, list)
        or len(value) > 100
        or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f-]{36}", v) for v in value)
    ):
        raise RagError(400, "VALIDATION_ERROR", f"Invalid {field}.")
    return list(dict.fromkeys(value))


def mode(value="hybrid"):
    if value not in ("baseline", "hybrid"):
        raise RagError(400, "VALIDATION_ERROR", "Invalid retrieval mode.")
    return value


async def body(request: Request, allowed):
    raw = await request.body()
    if len(raw) > 1024 * 1024:
        raise RagError(413, "VALIDATION_ERROR", "JSON body exceeds 1 MiB.")
    try:
        parsed = json.loads(raw)
    except (ValueError, UnicodeError):
        raise RagError(400, "VALIDATION_ERROR", "Malformed JSON body.") from None
    if not isinstance(parsed, dict):
        raise RagError(400, "VALIDATION_ERROR", "Expected a JSON object.")
    unknown = set(parsed) - set(allowed)
    if unknown:
        raise RagError(
            400, "VALIDATION_ERROR", "Unknown request fields.", {"fields": ", ".join(sorted(unknown))}
        )
    return parsed


def create_app(config=None, **options):
    config = config or load_config()
    service = RagService(config, **options)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await run_in_threadpool(service.close)

    app = FastAPI(title="PRO_RAG Studio", version="0.2.0", lifespan=lifespan)
    app.state.service = service

    @app.middleware("http")
    async def local_only(request, call_next):
        origin = request.headers.get("origin")
        if origin and origin not in ORIGINS:
            return problem(
                RagError(403, "ORIGIN_FORBIDDEN", "This local API only accepts the local dashboard origin.")
            )
        content_length = request.headers.get("content-length")
        maximum = (
            10 * config.upload_limit_bytes + 1024 * 1024
            if request.headers.get("content-type", "").startswith("multipart/")
            else 1024 * 1024
        )
        if content_length and content_length.isdigit() and int(content_length) > maximum:
            return problem(
                RagError(
                    413,
                    "FILE_TOO_LARGE" if maximum > 1024 * 1024 else "VALIDATION_ERROR",
                    "Request body exceeds the size limit.",
                )
            )
        response = Response(status_code=204) if request.method == "OPTIONS" else await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        if origin:
            response.headers.update(
                {
                    "Access-Control-Allow-Origin": origin,
                    "Vary": "Origin",
                    "Access-Control-Allow-Methods": "GET,POST,PATCH,DELETE,OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type",
                }
            )
        return response

    @app.exception_handler(RagError)
    async def rag_error(request, error):
        return problem(error)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        return problem(RagError(400, "VALIDATION_ERROR", "Invalid request."))

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return problem(
            RagError(
                error.status_code,
                "NOT_FOUND" if error.status_code == 404 else "VALIDATION_ERROR",
                "Route not found." if error.status_code == 404 else "Invalid request.",
            )
        )

    @app.exception_handler(Exception)
    async def internal_error(request, error):
        return problem(RagError(500, "INTERNAL_ERROR", "Unexpected server error."))

    @app.get(PREFIX + "/health")
    def health():
        return {"status": "ok", "version": "1", "time": timestamp(), "indexStatus": "ready", "playgroundVersion": 3}

    @app.get(PREFIX + "/config/status")
    def config_status():
        return service.config_status()

    @app.get(PREFIX + "/playground/options")
    async def playground_options():
        return service.playground.models.options()

    @app.post(PREFIX + "/playground/run")
    async def playground_run(request: PlaygroundRequest):
        return await run_in_threadpool(service.playground.run, request)

    @app.post(PREFIX + "/playground/stream")
    async def playground_stream(request: PlaygroundRequest):
        return StreamingResponse(
            stream_experiment(service.playground, request),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.get(PREFIX + "/collections")
    def collections(limit: str = "20", cursor: str | None = None):
        return service.list_collections(limit, cursor)

    @app.post(PREFIX + "/collections", status_code=201)
    async def create_collection(request: Request):
        data = await body(request, ["name", "description"])
        data = {
            "name": string(data.get("name"), "name", 1, 80),
            "description": string(data.get("description", ""), "description", 0, 500),
        }
        return await run_in_threadpool(service.create_collection, data)

    @app.patch(PREFIX + "/collections/{collection_id}")
    async def update_collection(collection_id: str, request: Request):
        data = await body(request, ["name", "description"])
        if not data:
            raise RagError(400, "VALIDATION_ERROR", "No update fields supplied.")
        patch = {
            key: string(value, key, 1 if key == "name" else 0, 80 if key == "name" else 500)
            for key, value in data.items()
        }
        return await run_in_threadpool(service.update_collection, collection_id, patch)

    @app.delete(PREFIX + "/collections/{collection_id}", status_code=204)
    def delete_collection(collection_id: str):
        service.delete_collection(collection_id)
        return Response(status_code=204)

    @app.post(PREFIX + "/documents", status_code=202)
    async def upload(request: Request):
        async with request.form(max_files=10, max_fields=2) as form:
            collection_id = string(form.get("collectionId"), "collectionId", 36, 36)
            ocr = form.get("ocr", "false")
            if ocr not in ("true", "false"):
                raise RagError(400, "VALIDATION_ERROR", "Invalid ocr flag.")
            files = []
            for key, item in form.multi_items():
                if key == "files" and hasattr(item, "read"):
                    content = await item.read(config.upload_limit_bytes + 1)
                    if len(content) > config.upload_limit_bytes:
                        raise RagError(413, "FILE_TOO_LARGE", "File exceeds 25 MiB.")
                    files.append(
                        {"name": item.filename or "", "content": content, "mimeType": item.content_type}
                    )
                elif key not in ("collectionId", "ocr"):
                    raise RagError(400, "VALIDATION_ERROR", "Invalid upload field.")
            return await run_in_threadpool(service.upload, collection_id, files, ocr == "true")

    @app.get(PREFIX + "/documents")
    def documents(collectionId: str, limit: str = "20", cursor: str | None = None, status: str | None = None):
        collection_id = string(collectionId, "collectionId", 36, 36)
        if status is not None and status not in ("queued", "processing", "ready", "failed"):
            raise RagError(400, "VALIDATION_ERROR", "Invalid status filter.")
        return service.list_documents(collection_id, limit, cursor, status)

    @app.get(PREFIX + "/documents/{document_id}")
    def document(document_id: str):
        return service.get_document(document_id)

    @app.get(PREFIX + "/documents/{document_id}/chunks")
    def chunks(document_id: str, limit: str = "20", cursor: str | None = None):
        return service.document_chunks(document_id, limit, cursor)

    @app.post(PREFIX + "/documents/{document_id}/retry", status_code=202)
    def retry(document_id: str):
        return service.retry_document(document_id)

    @app.delete(PREFIX + "/documents/{document_id}", status_code=204)
    def delete_document(document_id: str):
        service.delete_document(document_id)
        return Response(status_code=204)

    @app.post(PREFIX + "/queries")
    async def query(request: Request):
        data = await body(request, ["collectionId", "question", "documentIds", "mode", "topK", "generate"])
        parsed = {
            "collectionId": string(data.get("collectionId"), "collectionId", 36, 36),
            "question": string(data.get("question"), "question", 2, 2000),
            "documentIds": ids(data.get("documentIds"), "documentIds"),
            "mode": mode(data.get("mode", "hybrid")),
            "topK": count(data.get("topK"), "topK"),
            "generate": boolean(data.get("generate"), "generate"),
        }
        return await run_in_threadpool(service.query, parsed)

    @app.get(PREFIX + "/queries")
    def queries(limit: str = "20", cursor: str | None = None, collectionId: str | None = None):
        return service.list_queries(limit, cursor, collectionId)

    @app.get(PREFIX + "/queries/{query_id}")
    def query_record(query_id: str):
        result = service.store.get("query", query_id)
        if not result:
            raise RagError(404, "NOT_FOUND", "Query not found.")
        return result

    @app.get(PREFIX + "/stats")
    def stats(collectionId: str | None = None):
        return service.stats(collectionId)

    @app.get(PREFIX + "/evaluations/cases")
    def cases(limit: str = "20", cursor: str | None = None, collectionId: str | None = None):
        return service.store.page(service.store.all("eval_case", collectionId), limit, cursor)

    @app.post(PREFIX + "/evaluations/cases", status_code=201)
    async def create_case(request: Request):
        data = await body(request, ["collectionId", "question", "expectedDocumentIds", "answerable"])
        answerable = boolean(data.get("answerable"), "answerable")
        expected = ids(data.get("expectedDocumentIds"), "expectedDocumentIds")
        if answerable and not expected:
            raise RagError(400, "VALIDATION_ERROR", "Answerable cases need an expected document.")
        parsed = {
            "collectionId": string(data.get("collectionId"), "collectionId", 36, 36),
            "question": string(data.get("question"), "question", 2, 2000),
            "expectedDocumentIds": expected,
            "answerable": answerable,
        }
        return await run_in_threadpool(service.create_case, parsed)

    @app.delete(PREFIX + "/evaluations/cases/{case_id}", status_code=204)
    def delete_case(case_id: str):
        if not service.store.delete("eval_case", case_id):
            raise RagError(404, "NOT_FOUND", "Evaluation case not found.")
        return Response(status_code=204)

    @app.post(PREFIX + "/evaluations/runs", status_code=201)
    async def evaluate(request: Request):
        data = await body(request, ["collectionId", "mode", "topK"])
        parsed = {
            "collectionId": string(data.get("collectionId"), "collectionId", 36, 36),
            "mode": mode(data.get("mode")),
            "topK": count(data.get("topK"), "topK"),
        }
        return await run_in_threadpool(service.run_evaluation, parsed)

    @app.get(PREFIX + "/evaluations/runs")
    def runs(limit: str = "20", cursor: str | None = None, collectionId: str | None = None):
        return service.store.page(
            [service.run_summary(r) for r in service.store.all("eval_run", collectionId)], limit, cursor
        )

    @app.get(PREFIX + "/evaluations/runs/{run_id}")
    def run(run_id: str):
        result = service.store.get("eval_run", run_id)
        if not result:
            raise RagError(404, "NOT_FOUND", "Evaluation run not found.")
        return result

    return app
