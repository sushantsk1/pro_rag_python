# PRO_RAG architecture

The existing React dashboard communicates with a local **Python FastAPI** service. The backend keeps /api/v1 routes, camelCase responses, structured errors, and HTTP statuses expected by the dashboard.

~~~mermaid
flowchart LR
  UI[React dashboard] --> API[FastAPI]
  API --> Store[SQLite and original files]
  Store --> Parse[Isolated Python parsers]
  Parse --> Split[LangChain character splitter]
  Split --> Embed[HuggingFaceEmbeddings / MiniLM]
  Embed --> Store
  Store --> BM25[LangChain BM25]
  Store --> FAISS[LangChain FAISS]
  API --> BM25
  API --> FAISS
  BM25 --> RRF[Reciprocal rank fusion]
  FAISS --> RRF
  RRF --> Rank[Heuristic rerank / top K]
  Rank --> Chain[LangChain prompt / ChatNVIDIA / parser]
  Chain --> Cite[Citation validation]
  Cite --> UI
~~~

## Modules

| Module | Responsibility |
| --- | --- |
| server/main.py | Loopback Uvicorn entry point |
| server/app.py | API contract, validation, local CORS, sanitized errors |
| server/service.py | Collections, ingestion jobs, recovery, querying, evaluation |
| server/store.py | Thread-safe SQLite with the original records schema |
| server/parser.py and parse_worker.py | Parsing, structure-aware sections, splitting, optional OCR, subprocess timeout |
| server/embedding.py | LangChain Embeddings interface and lazy HuggingFaceEmbeddings |
| server/retrieval.py | LangChain Documents, BM25Retriever, FAISS, runnable retrieval |
| server/provider.py | ChatPromptTemplate / ChatNVIDIA / StrOutputParser / citation validation |

## Ingestion

Uploads preserve format/size checks, duplicate hashes, up to 10 files per request and 100 documents per collection. A single background executor processes jobs. Each parser runs in an isolated subprocess with a 45-second timeout, or 120 seconds for OCR. Successful chunk replacement is atomic. Failed documents expose retry; interrupted processing becomes failed on restart.

Format parsers use pypdf, python-docx, python-pptx, openpyxl, csv, and BeautifulSoup. LangChain RecursiveCharacterTextSplitter uses 900 characters, 140 overlap, and paragraph/newline/space/character separators. It splits within pages, slides, and heading sections. Markdown title/version preambles are carried into child sections; TXT chunks record starting lines. CSV/XLSX and Office tables use labeled rows; unusually long rows are split. Boundaries need not exactly match OfficeParser.

Optional OCR renders scanned PDF pages with pypdfium2 and recognizes text with pytesseract. It requires the Python OCR extra, external Tesseract, and language assets. Readiness is reported only after text was recognized.

## Embeddings and persistence

HuggingFaceEmbeddings runs sentence-transformers/all-MiniLM-L6-v2 locally on CPU with normalized output. Python Sentence Transformers weights replace the old quantized JS ONNX export. Models load lazily and are cached. The model is configurable; vector dimensions are inferred.

SQLite stores canonical chunks, vector arrays, metadata, IDs, query history, and evaluation records. Each new embedding records its model identity. Hybrid retrieval regenerates absent/mismatched embeddings, including unmarked JS vectors. Baseline remains available during an embedding outage. Test vectors are never substituted unless EMBEDDING_MODE=stub is explicit.

## Retrieval

Indexes are built lazily for the selected collection/document set, enforcing filters before candidate selection. A cache holds at most eight combinations; ingestion, deletion, and embedding changes invalidate it. FAISS uses exact inner-product search, equivalent to cosine similarity for normalized vectors. BM25 excludes chunks with no matching tokens; negative tiny-corpus scores are clamped to zero for the heuristic.

Hybrid takes up to max(4*K,24) candidates per channel. Reciprocal rank fusion with constant 60 retains max(3*K,18), then reranking selects K, default 6. Baseline skips the vector channel while retaining fusion/rerank stages.

Final score: 45% token coverage + 30% nonnegative semantic similarity + 20% clipped BM25 score/5 + 5% exact question phrase. This is a heuristic, not a cross-encoder or calibrated probability.

## Generation and evaluation

An evidence score below 0.3 triggers local abstention. Otherwise ChatPromptTemplate, ChatNVIDIA, StrOutputParser, and a citation validator form the answer chain. The default model remains nvidia/nemotron-3.5-lightning-30b-a3b. Key, endpoint, timeout, and thinking flag retain their environment settings. SDK aggregate usage telemetry is disabled.

A deterministic conflict heuristic can cite both claims and abstain. Explicit date/version questions are left to the model to apply source scope. Missing/invented citations or citation-only text cause abstention. Provider failures return sanitized codes without credentials.

Evaluation runs real retrieval and a local fixture answer path; it makes no NVIDIA request and does not grade live answer quality. Metrics include document hit rate at K, reciprocal rank, citation validity, abstention accuracy, and latency. Recall/MRR include answerable cases only. Case/corpus fingerprints support fair comparisons. JS evaluation runs remain historical.

## Operating boundary

Use one API worker and one user on loopback. A service lock serializes mutable retrieval work; SQLite uses WAL and transactions. Indexes are reconstructed from canonical data without pickle loading. No distributed queue/index or multi-user authentication is provided. Only the React frontend requires Node; backend, scripts, and tests are Python.
