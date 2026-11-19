# Testing the Python migration

Run commands from the repository root. Activate the virtual environment or use its interpreter explicitly.

## Offline checks

~~~powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check server scripts tests/python
.\.venv\Scripts\python.exe -m compileall -q server scripts
npm.cmd --prefix web run build
node --test tests/web/playground-stream.test.mjs
~~~

The default suite makes no hosted NVIDIA calls or model downloads. It covers the original API contract, a real Uvicorn subprocess, eight file formats, source locations, overlap, bad/duplicate uploads, filters, citations, abstention/conflicts, retries, embedding outage/recovery, evaluation fingerprints, deletion, demo seeding, and legacy SQLite across restarts.

FAISS/BM25 run with deterministic fixture embeddings. Provider tests execute the actual LangChain ChatNVIDIA chain against a localhost fake server, including 401/403/429/500 and timeout behavior.

The playground tests cover all four chunking strategies across three retrieval
modes and two embedding options, partial execution, vector normalization, source
offsets, parent context expansion, ranking order, prompt consistency, library
isolation, input limits, concurrent-run rejection, and generation failures. Local
model tests use explicit fixture vectors; TF–IDF, FAISS, and BM25 execute normally.

Projection tests compare the PCA geometry and explained variance against NumPy SVD,
check cosine scores against the full input vectors, ensure question changes leave
chunk coordinates unchanged, and cover one-point, identical-vector, rank-deficient,
and out-of-vocabulary cases. They also check the API links map points to indexed
chunk vectors and respects partial execution.

Streaming tests verify ordered stage events, early delivery while a real HTTP run's
embedding call is blocked, default generation, partial runs, provider errors,
disconnect checks, and retryable busy errors. Browser-reader tests cover fragmented
UTF-8/NDJSON, interrupted streams, server failures, and outdated backend messages.

A candidate-name regression runs the actual NVIDIA LangChain chain against the
localhost fake provider with a real normalized semantic hit, no literal query
overlap, and a low heuristic score. It checks that the provider receives the exact
prepared prompt and returns a cited name. Further tests distinguish no model call,
model abstention, and citation validation failures in the execution trace.

## Real MiniLM and FAISS

~~~powershell
$env:RAG_TEST_REAL_EMBEDDINGS = '1'
.\.venv\Scripts\python.exe -m pytest tests/python/test_real_embeddings.py -q -s
Remove-Item Env:RAG_TEST_REAL_EMBEDDINGS
~~~

This downloads/caches actual MiniLM weights on first use, checks normalized 384-dimensional vectors, ingests synthetic text, and verifies semantic scores through LangChain/FAISS. No NVIDIA request is made.

## Reproducible retrieval report

~~~powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_retrieval --repeats 5 --top-k 6
~~~

This runs the existing BM25 baseline and BM25 + FAISS hybrid pipeline on the same 16 synthetic
Markdown documents and 42 labeled questions, using real normalized MiniLM embeddings. It reports
document hit rate, document/passage recall, reciprocal rank, filter violations, the pre-generation
confidence threshold's decisions, and warm median/p95 retrieval latency. This is not a live-answer
or ChromaDB benchmark.

The script defaults to offline model loading; add `--allow-model-download` to permit the initial
Hugging Face download. It never substitutes stub embeddings, loads no `.env`, calls no NVIDIA
endpoint, and uses a temporary SQLite database separate from user collections. No running API or
frontend is required.

It replaces [the report](EVALUATION_REPORT.md) and [raw results](evaluation/results.json). Inspect
[the corpus and relevance labels](evaluation/dataset.json) alongside the results. Timing samples,
retrieved snippets, model revision, package versions, dataset/chunk fingerprints, and source hashes
are included for auditing. The dataset is a development set authored with its questions; the report
does not establish performance on independent or real-world data.

## Dashboard

Start python -m server.main and the React Vite server, then python -m scripts.seed_demo. Inspect Knowledge library, chunks, retrieval-only hybrid results, filters, trace, history, and both evaluation modes. Enable Generate answer only when you want excerpts sent to NVIDIA.

Open **RAG Playground** from the same sidebar. Verify changing chunk size changes
the real output after rerunning; try parent–child and semantic grouping, inspect
MiniLM/TF–IDF vectors and retrieval channels, and compare selected context. Verify
that full runs include generation without a checkbox, earlier-stage runs remain
local, old outputs are
marked after settings change, and the existing library remains unchanged. Test
algorithm selection by click and drag, completed-run playback, JSON export, and
desktop/tablet/mobile layouts. See [the feature guide](PLAYGROUND.md).

In the embedding map, verify point selection, full-vector cosine scores, retrieved
candidate highlights, 2D pan/zoom, 3D pointer and keyboard rotation, camera reset,
and complete chunk inspection. Pause flow and change playback speed; reduced-motion
preferences should disable moving packets. Inspect the tabbed panel and its close/
reopen behavior, and confirm exports include real projection coordinates. The
redesign was checked in local Chrome at 1920, 1366, 1024, 768, and 390 pixels using
cached real MiniLM embeddings and TF–IDF, without a hosted generation request.
Progress behavior was also checked with controlled embedding/provider delays:
only Embed runs, elapsed time advances, future stages cannot be selected, and the
map appears before generation finishes. The QA provider was explicitly a fixture;
the normal dashboard continues to use its configured NVIDIA provider.

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
