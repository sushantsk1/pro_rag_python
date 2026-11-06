# Python/LangChain migration — 6 November 2026

The JavaScript backend, CLI utilities, and backend tests are replaced by Python. The React dashboard and its REST contract are retained.

| Before | After |
| --- | --- |
| Express | FastAPI/Uvicorn |
| Transformers.js / Xenova MiniLM ONNX | LangChain HuggingFaceEmbeddings / Sentence Transformers MiniLM |
| Orama keyword/vector index | LangChain BM25Retriever + FAISS |
| OfficeParser chunking | Python format parsers + LangChain RecursiveCharacterTextSplitter |
| Direct NVIDIA fetch | LangChain ChatNVIDIA prompt and output-parser chain |
| Node SQLite | Python sqlite3, same canonical schema |
| Node test runner and CLI scripts | pytest and Python modules |

Compatibility covers collections, uploads across eight formats, source locations, failed-job retry, hybrid/baseline retrieval, collection/document filters, traces, cited answers, abstention/conflicts, query history, statistics, evaluation, and deletion. Existing SQLite records keep their IDs and metadata. Hybrid regenerates legacy embeddings instead of mixing Python and JS vectors. Scores and chunk boundaries are not expected to match exactly.

## Validation performed

- Offline Python suite: **34 passed**, with the optional real-model test skipped by default.
- Separate opt-in real MiniLM test: **1 passed**; normalized 384-dimensional vectors and semantic retrieval through FAISS.
- Actual Uvicorn entry point tested over localhost HTTP, including upload, ingestion, and a hybrid answer.
- Actual LangChain ChatNVIDIA chain tested against a local fake provider for valid answers, 401/403/429/500 errors, and timeout handling.
- Ruff lint/import checks pass; dependency consistency check passes.
- Existing React production build passes.

The backend migration has not been tested against a live NVIDIA account. Successful OCR extraction and visual browser inspection of the dashboard were not performed. The previous NVIDIA live report describes the historical JavaScript implementation only. See TESTING.md for reproducible checks and the optional synthetic live test.
