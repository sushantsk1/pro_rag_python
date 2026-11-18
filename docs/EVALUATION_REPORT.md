# Retrieval evaluation report

Measured at 2026-11-06T16:23:25.735456+05:30 (India Standard Time).

On this synthetic dataset, hybrid evidence recall@6 was 94.4%, versus 88.9% for the keyword baseline (+5.6 percentage points). Warm median retrieval time was 10.54 ms versus 1.52 ms. These measurements describe this corpus and machine, not general production performance.

## Scope

This measures the existing Python retrieval implementation with real local MiniLM embeddings. Baseline is BM25 followed by the existing fusion and heuristic reranking. Hybrid adds FAISS semantic retrieval before the same fusion and reranking. This is a pipeline comparison, not a FAISS-versus-ChromaDB benchmark. ChromaDB was not installed or measured.

NVIDIA was not called. Answer correctness, hallucination rate, live citation quality, and end-to-end generated-answer latency were not measured. The threshold results below only evaluate the existing confidence < 0.3 decision before generation; a model or conflict validator may subsequently abstain.

## Dataset and protocol

- 16 synthetic Markdown documents, 64 actual parsed chunks, 42 labeled questions (36 answerable, 6 unsupported).
- Topics include leave, expenses, travel, access control, security, backups, records retention, versioned warranties, returns, customer support, and procurement.
- Documents and questions were hand-authored together before measuring. Each answerable question has one or more expected document-and-quote labels, checked against the parsed chunks.
- Identical corpus, embeddings, filters, top K, and ranking code were used for both modes. Document filters restrict the candidate corpus before search.
- Top K = 6; chunks = 900 characters with 140 overlap; normalized 384-dimensional sentence-transformers/all-MiniLM-L6-v2 embeddings on CPU.
- One untimed pass per mode warms all question/filter combinations; 5 measured repetitions follow (210 timed queries per mode). Question order is shuffled with seed 20261106; mode order alternates.
- Modes have independent bounded index caches, preventing cross-mode cache eviction. Every measured result is checked against its warmup result for stability.
- Timings include retrieval's SQLite reads, BM25, query embedding for hybrid, FAISS, fusion, and reranking. They exclude HTTP, service-level embedding checks/locking, query-history writes, ingestion, model initialization, and answer generation.
- Warm p95 uses the nearest-rank percentile. Index-cold measurements below are one first-question sample per mode with the model already loaded; they are not full process/OS cold-start benchmarks.
- A temporary isolated SQLite database and synthetic uploads were used; existing user collections were not read or modified. No API credentials or .env settings were loaded.

## Results

| Metric | Keyword baseline | Hybrid |
| --- | ---: | ---: |
| Document hit rate@6 | 94.4% | 97.2% |
| Document MRR@6 | 0.887 | 0.944 |
| Document recall@6 | 94.4% | 97.2% |
| Evidence recall@6 | 88.9% | 94.4% |
| All required evidence found@6 | 88.9% | 94.4% |
| First-evidence MRR@6 | 0.802 | 0.853 |
| Pre-generation threshold decision accuracy | 66.7% | 88.1% |
| Unsupported questions blocked by threshold | 33.3% | 33.3% |
| Answerable questions incorrectly blocked | 27.8% | 2.8% |
| Filtered-case source violations | 0 | 0 |
| Warm median retrieval (ms) | 1.52 | 10.54 |
| Warm p95 retrieval (ms) | 2.20 | 16.03 |

Metric definitions:

- Document hit rate@6: fraction of answerable cases with at least one expected document among the top 6 chunks. This is the dashboard's existing `recallAtK` metric.
- Document recall: fraction of distinct expected documents found per case, averaged across answerable cases. Unlike hit rate, multi-document cases require both documents for full credit.
- Evidence recall: fraction of labeled document-and-quote targets found in returned snippets per case, averaged across answerable cases. Finding the correct document's wrong section does not count as finding the evidence.
- All required evidence found: fraction of answerable cases with every labeled quote retrieved.
- MRR: mean reciprocal rank of the first matching chunk; missing matches score zero. For multi-source questions, first-evidence MRR measures the first target found, not completeness.
- Unsupported cases are excluded from retrieval relevance denominators. Threshold decision accuracy includes all questions; unsupported abstention and answerable false-abstention rates are also reported separately to expose class imbalance.

## Results by question category

| Category | Questions | Baseline evidence recall | Hybrid evidence recall | Baseline first-evidence MRR | Hybrid first-evidence MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct | 14 | 100.0% | 100.0% | 1.000 | 0.964 |
| filtered | 4 | 100.0% | 100.0% | 1.000 | 1.000 |
| multi-source | 2 | 100.0% | 100.0% | 1.000 | 1.000 |
| paraphrase | 14 | 71.4% | 85.7% | 0.526 | 0.657 |
| unanswerable | 6 | n/a | n/a | n/a | n/a |
| version | 2 | 100.0% | 100.0% | 0.750 | 1.000 |

## Retrieval misses and ranking differences

Evidence ranks are per labeled target; `missing` means the required quote was absent.

| ID | Question | Baseline evidence ranks | Hybrid evidence ranks |
| --- | --- | --- | --- |
| Q10 | How much can I spend on food each day during a work trip within the country? | missing | 1 |
| Q12 | When must I be available online while working from home? | 3 | 2 |
| Q14 | How quickly must IT revoke a worker's logins after they leave the company? | missing | missing |
| Q16 | Do I need to regularly change my passphrase if it has not been stolen? | 5 | 2 |
| Q17 | Within how many minutes must a suspected security incident be reported? | 1 | 2 |
| Q20 | What is the maximum acceptable downtime when restoring the database? | missing | 5 |
| Q23 | How long is the current hardware warranty for purchases after September 1, 2026? | 2 | 1 |
| Q26 | How long do I have to send back a product that arrived broken after I opened it? | 3 | 2 |
| Q28 | Can I lodge a help request through the website in the middle of the night? | missing | missing |

## Unsupported questions

A high similarity score can occur for an unsupported question that resembles a policy. Passing the threshold means generation would be allowed, not that a correct answer exists.

| ID | Question | Baseline score / threshold action | Hybrid score / threshold action |
| --- | --- | --- | --- |
| Q37 | What is the annual leave policy for contractors? | 0.538 / allow generation | 0.737 / allow generation |
| Q38 | What is the international hotel reimbursement cap in London? | 0.470 / allow generation | 0.689 / allow generation |
| Q39 | Does the hardware warranty cover battery capacity degradation? | 0.325 / allow generation | 0.475 / allow generation |
| Q40 | What is the chief financial officer's personal phone number? | 0.425 / allow generation | 0.500 / allow generation |
| Q41 | Which GPU architecture powers the production inference servers? | 0.228 / block | 0.277 / block |
| Q42 | What is the office cafeteria's lunch menu today? | 0.209 / block | 0.262 / block |

## Setup and environment

- Platform: Windows-11-10.0.26200-SP0; Python 3.13.14.
- CPU: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel; 22 logical CPUs; 31.46 GiB physical RAM; PyTorch intra-operation threads: 16.
- Cached-model initialization and one probe: 10.51 seconds.
- Parsing, document embedding, and storage: 15.73 seconds with the model already loaded.
- Index-cold / model-warm first retrieval: baseline 29.88 ms; hybrid 91.17 ms.
- Dataset SHA-256: `72ce4a1f002f758e51e09c3bb291054bd19a2e9508711829bb38a4f7889bfe21`.
- Canonical parsed chunks and embedding hashes SHA-256: `110e29d97bc439241d7a579006a354a3e8db2f04a9307899b8d1b6f1b9fa91b9`.
- Cached model refs: `{"main": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"}`.

| Package | Version |
| --- | --- |
| langchain | 1.4.9 |
| langchain-community | 0.4.2 |
| langchain-huggingface | 1.2.3 |
| sentence-transformers | 5.7.0 |
| torch | 2.15.0 |
| faiss-cpu | 1.15.1 |
| rank-bm25 | 0.2.2 |
| numpy | 2.5.4 |

## Interpretation and limits

- Use the observed evidence recall and latency together. A hybrid pipeline performs extra embedding/search work and is not expected to be a universal latency improvement.
- Inspect the per-question failures and category breakdown before claiming hybrid is better. Retrieving the right document is weaker evidence than retrieving the answer passage.
- This is a small, synthetic development set authored with the corpus, without independent annotators, a held-out test set, or statistical significance analysis. Do not generalize these percentages to real users, other languages, or larger corpora.
- All inputs are Markdown and use the same parser/chunker as uploads. This does not benchmark PDF/Office parsing quality or scanned-document OCR.
- Quote labels identify expected evidence, not every potentially relevant passage. Version and multi-source cases test retrieval, not whether a model interprets dates or resolves conflicts.
- No ranking weights, chunking parameters, or confidence threshold were tuned after observing results.
- Latency is from one local process, without a controlled idle machine, concurrency load, confidence intervals, or production network conditions. Memory, disk, and ChromaDB performance comparisons are outside this run.

## Reproduce

From the repository root, with the existing dependencies and cached MiniLM model:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_retrieval --repeats 5 --top-k 6
```

No virtual environment activation, running API/frontend, or NVIDIA key is required. The script runs offline by default and fails if real embeddings are unavailable; it never substitutes stub vectors. On a new machine, add `--allow-model-download` for the initial Hugging Face download.

Artifacts: [dataset and labels](evaluation/dataset.json), [raw results, environment, source hashes, retrieved snippets, and timing samples](evaluation/results.json), [runner](../scripts/evaluate_retrieval.py). Re-running replaces this report and results.json.
