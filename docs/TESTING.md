# Testing the Python migration

Run commands from the repository root. Activate the virtual environment or use its interpreter explicitly.

## Offline checks

~~~powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check server scripts tests/python
.\.venv\Scripts\python.exe -m compileall -q server scripts
npm.cmd --prefix web run build
~~~

The default suite makes no hosted NVIDIA calls or model downloads. It covers the original API contract, a real Uvicorn subprocess, eight file formats, source locations, overlap, bad/duplicate uploads, filters, citations, abstention/conflicts, retries, embedding outage/recovery, evaluation fingerprints, deletion, demo seeding, and legacy SQLite across restarts.

FAISS/BM25 run with deterministic fixture embeddings. Provider tests execute the actual LangChain ChatNVIDIA chain against a localhost fake server, including 401/403/429/500 and timeout behavior.

## Real MiniLM and FAISS

~~~powershell
$env:RAG_TEST_REAL_EMBEDDINGS = '1'
.\.venv\Scripts\python.exe -m pytest tests/python/test_real_embeddings.py -q -s
Remove-Item Env:RAG_TEST_REAL_EMBEDDINGS
~~~

This downloads/caches actual MiniLM weights on first use, checks normalized 384-dimensional vectors, ingests synthetic text, and verifies semantic scores through LangChain/FAISS. No NVIDIA request is made.

## Dashboard

Start python -m server.main and the React Vite server, then python -m scripts.seed_demo. Inspect Knowledge library, chunks, retrieval-only hybrid results, filters, trace, history, and both evaluation modes. Enable Generate answer only when you want excerpts sent to NVIDIA.

## Optional live NVIDIA test

Start an isolated backend:

~~~powershell
$env:PORT = '8795'
$env:DATA_DIR = './data/live-python'
$env:MODEL_MODE = 'nvidia'
$env:EMBEDDING_MODE = 'local'
.\.venv\Scripts\python.exe -m server.main
~~~

In a second terminal:

~~~powershell
$env:RAG_BASE_URL = 'http://127.0.0.1:8795/api/v1'
.\.venv\Scripts\python.exe -m scripts.seed_demo
$env:RAG_LIVE_TEST = '1'
.\.venv\Scripts\python.exe -m scripts.live_test
~~~

The script verifies the synthetic corpus against its file hashes, checks four generated answers and two abstentions, and validates citations/CSV row locations. It needs NVIDIA access/quota and does not retry automatically. The historical [JavaScript live report](LIVE_TEST_REPORT.md) is not evidence of a Python live run.

## OCR and fixtures

Optional OCR requires pip install -e ".[ocr]", external Tesseract, and language assets. The default suite checks scanned-PDF/no-text failure; successful OCR needs a separate check on the target machine.

Original binary fixtures remain under tests/integration/fixtures. Python -m scripts.generate_fixtures regenerates equivalent fixtures using test dependencies. This command intentionally rewrites those fixture files; binary bytes can differ from the old generator.
