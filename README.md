# PRO_RAG Studio

A local document analysis application with a **Python + LangChain backend** and the existing **React dashboard**. Upload documents, inspect chunks, compare keyword and hybrid retrieval, and generate NVIDIA answers with source citations.

Designed for one user on one computer and a portfolio-sized corpus. Parsing, storage, embeddings, retrieval, and evaluation run locally. Generation sends the question and selected excerpts to NVIDIA.

## Stack

| Responsibility | Implementation |
| --- | --- |
| API | FastAPI + Uvicorn, same /api/v1 contract, port 8787 |
| Embeddings | LangChain HuggingFaceEmbeddings, sentence-transformers/all-MiniLM-L6-v2, normalized 384-dimensional CPU vectors |
| Vector search | LangChain FAISS; normalized inner product (cosine similarity) |
| Keyword search | LangChain BM25Retriever with rank-bm25 |
| Retrieval | Keyword + semantic, reciprocal rank fusion, heuristic reranking, default top 6 |
| Chunking | LangChain recursive character splitter, 900 characters / 140 overlap inside structured sections; labeled table rows |
| Generation | LangChain ChatNVIDIA prompt/output-parser chain; same configurable NVIDIA model |
| Storage | SQLite data/rag.sqlite and original files in data/files |
| Dashboard | Existing React/Vite interface on port 5173 |

## Setup on Windows

Requires **Python 3.11–3.14** and **Node.js 20.19+** for the dashboard. The validated environment uses Python 3.13. Initial installation and model downloads need internet access.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
Copy-Item .env.example .env
~~~

Put your NVIDIA key in .env for generated answers. Uploads, retrieval-only queries, and offline tests need no key. Environment variables override .env; .ENV is also supported. Keep credentials private.

Terminal 1 — backend:

~~~powershell
.\.venv\Scripts\python.exe -m server.main
~~~

Terminal 2 — dashboard:

~~~powershell
Set-Location web
npm.cmd ci
npm.cmd run dev
~~~

Open <http://127.0.0.1:5173>. API documentation: <http://127.0.0.1:8787/docs>.

On Linux/macOS use .venv/bin/python and npm. Alternatively activate the virtual environment and use python; root npm start, npm test, and npm run seed delegate to Python.

For backend auto-reload:

~~~powershell
.\.venv\Scripts\python.exe -m uvicorn server.app:create_app --factory --reload --host 127.0.0.1 --port 8787
~~~

## Demo and checks

With the backend running:

~~~powershell
.\.venv\Scripts\python.exe -m scripts.seed_demo
~~~

This idempotently loads three synthetic files from docs/demo and three evaluation cases without generating NVIDIA answers. Open Northstar Support, ask about the warranty, inspect citations, and compare hybrid/baseline.

~~~powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check server scripts tests/python
npm.cmd --prefix web run build
~~~

See [testing](docs/TESTING.md), [architecture](docs/ARCHITECTURE.md), and the [beginner's guide](docs/BEGINNERS_GUIDE.md).

For the exact validated dependency versions, install requirements.lock.txt before the editable project:

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
~~~

## Existing JavaScript data

Python keeps the SQLite schema and camelCase API responses. Collections, files, document/chunk IDs, and history remain readable. Keep DATA_DIR pointed at the existing directory, and stop the old Node API before starting Python on the same port/data directory.

Orama is replaced by FAISS + BM25, reconstructed from SQLite and cached in RAM. Legacy ONNX vectors are regenerated on the next hybrid query; baseline does not need embeddings. The old Xenova/all-MiniLM-L6-v2 setting maps to the Python Sentence Transformers model. Each new embedding records its model identity to prevent mixing vector spaces when models change. FAISS infers dimensions.

Scores, rankings, and chunk boundaries can differ from Orama/OfficeParser. Old evaluation runs remain historical records; rerun both modes for a fair Python comparison.

## Limits

Use one API worker on loopback. This application has no multi-user authentication or distributed index/job service.

Supported formats: PDF, DOCX, PPTX, XLSX, CSV, TXT, Markdown, HTML. Available page, slide, heading, sheet, and row locations are retained. Complex layouts, handwriting, protected files, and embedded images may need preprocessing.

Optional scanned-PDF OCR requires pip install -e ".[ocr]", a separate Tesseract installation, and language assets. Missing dependencies produce an actionable error.

MiniLM is intended for short English passages. Character-based chunking does not guarantee every chunk fits the model's token limit. Reranking/conflict detection are heuristics; valid citations do not certify factual accuracy. NVIDIA availability, quota, and model access depend on your account.
