"""Reproduce a local, real-embedding retrieval report without calling an LLM."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import sys
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

from server.config import Config
from server.embedding import LocalEmbedder
from server.retrieval import RetrievalEngine
from server.service import RagService

ROOT = Path(__file__).resolve().parents[1]
MODES = ("baseline", "hybrid")
THRESHOLD = 0.3  # The service's existing pre-generation evidence threshold.
SEED = 20261106


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalized(value):
    return " ".join(value.casefold().split())


def validate_dataset(dataset):
    documents = {doc["name"]: doc["text"] for doc in dataset["documents"]}
    if len(documents) != len(dataset["documents"]):
        raise ValueError("Duplicate document names")
    ids = [case["id"] for case in dataset["cases"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Cases must have unique IDs and must not be empty")
    for case in dataset["cases"]:
        filters = case.get("filterDocuments", [])
        if any(name not in documents for name in filters):
            raise ValueError(f"Unknown filter in {case['id']}")
        for target in case["evidence"]:
            if not target["quote"] or normalized(target["quote"]) not in normalized(
                documents[target["document"]]
            ):
                raise ValueError(f"Missing evidence quote in {case['id']}")
            if filters and target["document"] not in filters:
                raise ValueError(f"Evidence excluded by filter in {case['id']}")


def score_case(case, retrieved):
    sources = retrieved["sources"]
    expected = {target["document"] for target in case["evidence"]}
    document_ranks = [i for i, source in enumerate(sources, 1) if source["documentName"] in expected]
    evidence_ranks = [
        next(
            (
                i
                for i, source in enumerate(sources, 1)
                if source["documentName"] == target["document"]
                and normalized(target["quote"]) in normalized(source["snippet"])
            ),
            None,
        )
        for target in case["evidence"]
    ]
    present = [rank for rank in evidence_ranks if rank is not None]
    answerable = bool(expected)
    would_abstain = retrieved["confidence"] < THRESHOLD
    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "filterDocuments": case.get("filterDocuments", []),
        "evidence": case["evidence"],
        "answerable": answerable,
        "documentHit": bool(document_ranks) if answerable else None,
        "documentReciprocalRank": 1 / document_ranks[0] if document_ranks else 0,
        "documentRecall": len(expected & {s["documentName"] for s in sources}) / len(expected)
        if answerable
        else None,
        "evidenceRecall": len(present) / len(evidence_ranks) if answerable else None,
        "allEvidenceFound": len(present) == len(evidence_ranks) if answerable else None,
        "evidenceReciprocalRank": 1 / min(present) if present else 0,
        "evidenceRanks": evidence_ranks,
        "confidence": retrieved["confidence"],
        "wouldAbstain": would_abstain,
        "thresholdDecisionCorrect": would_abstain != answerable,
        "filterViolationCount": sum(
            source["documentName"] not in case["filterDocuments"] for source in sources
        )
        if case.get("filterDocuments")
        else 0,
        "sources": [
            {
                "document": source["documentName"],
                "location": source["location"],
                "snippet": source["snippet"],
                "scores": source["scores"],
            }
            for source in sources
        ],
    }


def percentile(values, fraction):
    """Nearest-rank percentile: sorted(values)[ceil(p*n)-1]."""
    return sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)]


def summarize(rows, samples):
    answerable = [row for row in rows if row["answerable"]]
    unsupported = [row for row in rows if not row["answerable"]]

    def mean(field, selected):
        return statistics.mean(float(row[field]) for row in selected) if selected else None

    latencies = [sample["durationMs"] for sample in samples]
    return {
        "caseCount": len(rows),
        "answerableCount": len(answerable),
        "unanswerableCount": len(unsupported),
        "documentHitRate": mean("documentHit", answerable),
        "documentMRR": mean("documentReciprocalRank", answerable),
        "documentRecall": mean("documentRecall", answerable),
        "evidenceRecall": mean("evidenceRecall", answerable),
        "allEvidenceRate": mean("allEvidenceFound", answerable),
        "evidenceMRR": mean("evidenceReciprocalRank", answerable),
        "thresholdDecisionAccuracy": mean("thresholdDecisionCorrect", rows),
        "unsupportedAbstentionRate": mean("wouldAbstain", unsupported),
        "answerableFalseAbstentionRate": mean("wouldAbstain", answerable),
        "filterViolationCount": sum(row["filterViolationCount"] for row in rows),
        "warmMedianMs": statistics.median(latencies) if latencies else None,
        "warmP95Ms": percentile(latencies, 0.95) if latencies else None,
        "warmMeanMs": statistics.mean(latencies) if latencies else None,
        "timedQueryCount": len(latencies),
    }


def machine_info():
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "logicalCpuCount": os.cpu_count(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "langchain",
                "langchain-community",
                "langchain-huggingface",
                "sentence-transformers",
                "torch",
                "faiss-cpu",
                "rank-bm25",
                "numpy",
            )
        },
    }
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", wintypes.DWORD),
                ("load", wintypes.DWORD),
                ("totalPhysical", ctypes.c_ulonglong),
                ("availablePhysical", ctypes.c_ulonglong),
                ("totalPage", ctypes.c_ulonglong),
                ("availablePage", ctypes.c_ulonglong),
                ("totalVirtual", ctypes.c_ulonglong),
                ("availableVirtual", ctypes.c_ulonglong),
                ("availableExtended", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            info["physicalRamGiB"] = round(status.totalPhysical / 1024**3, 2)
    return info


def run_benchmark(dataset, *, repeats, top_k, allow_download=False):
    validate_dataset(dataset)
    if not allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model_cache = ROOT / "data" / "models"
    run_root = ROOT / "data" / "evaluation-runs"
    run_root.mkdir(parents=True, exist_ok=True)
    print("Loading real cached MiniLM model (no NVIDIA calls)...", flush=True)
    with tempfile.TemporaryDirectory(prefix="retrieval-", dir=run_root) as work_dir:
        config = Config(
            data_dir=Path(work_dir),
            model_cache_dir=model_cache,
            embedding_mode="local",
            model_mode="stub",
        )
        embedder = LocalEmbedder(config)
        started = perf_counter()
        probe = embedder.embed_query("Local retrieval benchmark readiness check.")
        model_load_ms = (perf_counter() - started) * 1000
        if len(probe) != 384 or not math.isclose(sum(x * x for x in probe), 1, abs_tol=1e-4):
            raise RuntimeError("Expected real normalized 384-dimensional MiniLM embeddings")
        service = RagService(config, embedder=embedder)
        try:
            collection = service.create_collection(
                {"name": dataset["name"], "description": "Isolated benchmark"}
            )
            cid = collection["id"]
            started = perf_counter()
            for start in range(0, len(dataset["documents"]), 10):
                service.upload(
                    cid,
                    [
                        {"name": doc["name"], "content": doc["text"].encode()}
                        for doc in dataset["documents"][start : start + 10]
                    ],
                )
            service.wait_for_jobs()
            documents = service.store.all("document", cid)
            if any(doc["status"] != "ready" for doc in documents):
                raise RuntimeError("Benchmark document ingestion failed")
            service.ensure_embeddings(cid)
            ingestion_ms = (perf_counter() - started) * 1000
            chunks = service.store.all("chunk", cid)
            if not chunks or any(chunk.get("embeddingModel") != embedder.identity for chunk in chunks):
                raise RuntimeError("Missing or mismatched real embeddings")
            if any(
                len(chunk["embedding"]) != 384
                or not math.isclose(sum(x * x for x in chunk["embedding"]), 1, abs_tol=1e-4)
                for chunk in chunks
            ):
                raise RuntimeError("Invalid document embedding dimensions or normalization")
            by_name = {doc["name"]: doc["id"] for doc in documents}
            names_by_id = {doc["id"]: doc["name"] for doc in documents}
            canonical_chunks = [
                {
                    "document": names_by_id[c["documentId"]],
                    "index": c["index"],
                    "text": c["text"],
                    "location": c["location"],
                    "embeddingSha256": digest(c["embedding"]),
                }
                for c in chunks
            ]
            # Every expected quote must survive the actual parser/chunker.
            for case in dataset["cases"]:
                for target in case["evidence"]:
                    if not any(
                        c["document"] == target["document"]
                        and normalized(target["quote"]) in normalized(c["text"])
                        for c in canonical_chunks
                    ):
                        raise RuntimeError(f"Evidence label lost during chunking: {case['id']}")
            print(
                f"Ingested {len(documents)} documents / {len(chunks)} chunks with real embeddings.",
                flush=True,
            )
            # Independent caches prevent the modes evicting each other's filter indexes.
            engines = {mode: RetrievalEngine(service.store, embedder) for mode in MODES}

            def retrieve(mode, case):
                return engines[mode].retrieve(
                    collectionId=cid,
                    question=case["question"],
                    mode=mode,
                    topK=top_k,
                    documentIds=[by_name[name] for name in case.get("filterDocuments", [])],
                )

            cold = {}
            for mode in MODES:
                started = perf_counter()
                retrieve(mode, dataset["cases"][0])
                cold[mode] = (perf_counter() - started) * 1000
            outcomes = {mode: [] for mode in MODES}
            for mode in MODES:
                for case in dataset["cases"]:
                    outcomes[mode].append(score_case(case, retrieve(mode, case)))
                expected_cache_count = len(
                    {tuple(sorted(c.get("filterDocuments", []))) for c in dataset["cases"]}
                )
                if len(engines[mode].cache) != expected_cache_count:
                    raise RuntimeError(
                        "Filter combinations exceed the retrieval cache; warm measurements invalid"
                    )
            samples = {mode: [] for mode in MODES}
            rng = random.Random(SEED)
            for repetition in range(repeats):
                order = list(range(len(dataset["cases"])))
                rng.shuffle(order)
                for i in order:
                    case = dataset["cases"][i]
                    for mode in MODES if (repetition + i) % 2 == 0 else tuple(reversed(MODES)):
                        started = perf_counter()
                        result = retrieve(mode, case)
                        duration_ms = (perf_counter() - started) * 1000
                        check = score_case(case, result)
                        if check != outcomes[mode][i]:
                            raise RuntimeError(f"Non-deterministic result for {mode}/{case['id']}")
                        samples[mode].append(
                            {"caseId": case["id"], "repetition": repetition + 1, "durationMs": duration_ms}
                        )
                print(f"Completed measured repetition {repetition + 1}/{repeats}.", flush=True)
            summary = {mode: summarize(outcomes[mode], samples[mode]) for mode in MODES}
            categories = {
                category: {
                    mode: summarize([row for row in outcomes[mode] if row["category"] == category], [])
                    for mode in MODES
                }
                for category in sorted({case["category"] for case in dataset["cases"]})
            }
            model_refs = {}
            model_dir = model_cache / "models--sentence-transformers--all-MiniLM-L6-v2"
            if (model_dir / "refs").exists():
                model_refs = {
                    p.name: p.read_text().strip() for p in (model_dir / "refs").iterdir() if p.is_file()
                }
            source_files = [
                "scripts/evaluate_retrieval.py",
                "server/retrieval.py",
                "server/embedding.py",
                "server/parser.py",
                "server/parse_worker.py",
                "server/service.py",
                "server/store.py",
            ]
            import torch

            return {
                "generatedAt": datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat(),
                "dataset": {
                    "name": dataset["name"],
                    "version": dataset["version"],
                    "sha256": digest(dataset),
                    "categoryCounts": dict(Counter(c["category"] for c in dataset["cases"])),
                },
                "configuration": {
                    "topK": top_k,
                    "repeats": repeats,
                    "seed": SEED,
                    "embeddingModel": config.embedding_model,
                    "embeddingIdentity": embedder.identity,
                    "modelRefs": model_refs,
                    "dimensions": len(probe),
                    "device": "cpu",
                    "torchThreads": torch.get_num_threads(),
                    "chunkSize": 900,
                    "chunkOverlap": 140,
                    "abstentionThreshold": THRESHOLD,
                    "generationMeasured": False,
                    "allowModelDownload": allow_download,
                },
                "machine": machine_info(),
                "sourceSha256": {
                    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files
                },
                "corpus": {
                    "documentCount": len(documents),
                    "chunkCount": len(chunks),
                    "textBytes": sum(len(doc["text"].encode()) for doc in dataset["documents"]),
                    "chunksSha256": digest(canonical_chunks),
                    "chunks": canonical_chunks,
                },
                "setup": {
                    "modelInitializationMs": model_load_ms,
                    "ingestionMs": ingestion_ms,
                    "indexColdModelWarmMs": cold,
                },
                "summary": summary,
                "categories": categories,
                "outcomes": outcomes,
                "latencySamples": samples,
            }
        finally:
            service.close()


def render_report(result):
    baseline, hybrid = (result["summary"][mode] for mode in MODES)
    config, corpus, machine = result["configuration"], result["corpus"], result["machine"]
    k = config["topK"]

    def pct(value):
        return "n/a" if value is None else f"{value:.1%}"

    def number(value):
        return "n/a" if value is None else f"{value:.3f}"

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    evidence_delta = (hybrid["evidenceRecall"] - baseline["evidenceRecall"]) * 100
    lines = [
        "# Retrieval evaluation report",
        "",
        f"Measured at {result['generatedAt']} (India Standard Time).",
        "",
        f"On this synthetic dataset, hybrid evidence recall@{k} was {pct(hybrid['evidenceRecall'])}, "
        f"versus {pct(baseline['evidenceRecall'])} for the keyword baseline "
        f"({evidence_delta:+.1f} percentage points). Warm median retrieval time was "
        f"{hybrid['warmMedianMs']:.2f} ms versus {baseline['warmMedianMs']:.2f} ms. "
        "These measurements describe this corpus and machine, not general production performance.",
        "",
        "## Scope",
        "",
        "This measures the existing Python retrieval implementation with real local MiniLM embeddings. "
        "Baseline is BM25 followed by the existing fusion and heuristic reranking. Hybrid adds FAISS "
        "semantic retrieval before the same fusion and reranking. This is a pipeline comparison, "
        "not a FAISS-versus-ChromaDB benchmark. ChromaDB was not installed or measured.",
        "",
        "NVIDIA was not called. Answer correctness, hallucination rate, live citation quality, and "
        "end-to-end generated-answer latency were not measured. The threshold results below only "
        "evaluate the existing confidence < 0.3 decision before generation; a model or conflict "
        "validator may subsequently abstain.",
        "",
        "## Dataset and protocol",
        "",
        f"- {corpus['documentCount']} synthetic Markdown documents, {corpus['chunkCount']} actual parsed chunks, "
        f"{baseline['caseCount']} labeled questions ({baseline['answerableCount']} answerable, "
        f"{baseline['unanswerableCount']} unsupported).",
        "- Topics include leave, expenses, travel, access control, security, backups, records retention, "
        "versioned warranties, returns, customer support, and procurement.",
        "- Documents and questions were hand-authored together before measuring. Each answerable "
        "question has one or more expected document-and-quote labels, checked against the parsed chunks.",
        "- Identical corpus, embeddings, filters, top K, and ranking code were used for both modes. "
        "Document filters restrict the candidate corpus before search.",
        f"- Top K = {k}; chunks = 900 characters with 140 overlap; normalized 384-dimensional "
        "sentence-transformers/all-MiniLM-L6-v2 embeddings on CPU.",
        f"- One untimed pass per mode warms all question/filter combinations; {config['repeats']} measured "
        f"repetitions follow ({baseline['timedQueryCount']} timed queries per mode). "
        f"Question order is shuffled with seed {config['seed']}; mode order alternates.",
        "- Modes have independent bounded index caches, preventing cross-mode cache eviction. "
        "Every measured result is checked against its warmup result for stability.",
        "- Timings include retrieval's SQLite reads, BM25, query embedding for hybrid, FAISS, fusion, "
        "and reranking. They exclude HTTP, service-level embedding checks/locking, query-history "
        "writes, ingestion, model initialization, and answer generation.",
        "- Warm p95 uses the nearest-rank percentile. Index-cold measurements below are one first-question "
        "sample per mode with the model already loaded; they are not full process/OS cold-start benchmarks.",
        "- A temporary isolated SQLite database and synthetic uploads were used; existing user collections "
        "were not read or modified. No API credentials or .env settings were loaded.",
        "",
        "## Results",
        "",
        "| Metric | Keyword baseline | Hybrid |",
        "| --- | ---: | ---: |",
    ]
    metrics = [
        (f"Document hit rate@{k}", "documentHitRate", pct),
        (f"Document MRR@{k}", "documentMRR", number),
        (f"Document recall@{k}", "documentRecall", pct),
        (f"Evidence recall@{k}", "evidenceRecall", pct),
        (f"All required evidence found@{k}", "allEvidenceRate", pct),
        (f"First-evidence MRR@{k}", "evidenceMRR", number),
        ("Pre-generation threshold decision accuracy", "thresholdDecisionAccuracy", pct),
        ("Unsupported questions blocked by threshold", "unsupportedAbstentionRate", pct),
        ("Answerable questions incorrectly blocked", "answerableFalseAbstentionRate", pct),
        ("Filtered-case source violations", "filterViolationCount", str),
        ("Warm median retrieval (ms)", "warmMedianMs", lambda n: f"{n:.2f}"),
        ("Warm p95 retrieval (ms)", "warmP95Ms", lambda n: f"{n:.2f}"),
    ]
    lines += [f"| {name} | {fmt(baseline[field])} | {fmt(hybrid[field])} |" for name, field, fmt in metrics]
    lines += [
        "",
        "Metric definitions:",
        "",
        f"- Document hit rate@{k}: fraction of answerable cases with at least one expected document "
        f"among the top {k} chunks. This is the dashboard's existing `recallAtK` metric.",
        "- Document recall: fraction of distinct expected documents found per case, averaged across "
        "answerable cases. Unlike hit rate, multi-document cases require both documents for full credit.",
        "- Evidence recall: fraction of labeled document-and-quote targets found in returned snippets "
        "per case, averaged across answerable cases. Finding the correct document's wrong section "
        "does not count as finding the evidence.",
        "- All required evidence found: fraction of answerable cases with every labeled quote retrieved.",
        "- MRR: mean reciprocal rank of the first matching chunk; missing matches score zero. "
        "For multi-source questions, first-evidence MRR measures the first target found, not completeness.",
        "- Unsupported cases are excluded from retrieval relevance denominators. Threshold decision "
        "accuracy includes all questions; unsupported abstention and answerable false-abstention "
        "rates are also reported separately to expose class imbalance.",
        "",
        "## Results by question category",
        "",
        "| Category | Questions | Baseline evidence recall | Hybrid evidence recall | "
        "Baseline first-evidence MRR | Hybrid first-evidence MRR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category, modes in result["categories"].items():
        b, h = (modes[mode] for mode in MODES)
        lines.append(
            f"| {category} | {b['caseCount']} | {pct(b['evidenceRecall'])} | "
            f"{pct(h['evidenceRecall'])} | "
            f"{number(b['evidenceMRR']) if b['answerableCount'] else 'n/a'} | "
            f"{number(h['evidenceMRR']) if h['answerableCount'] else 'n/a'} |"
        )
    lines += ["", "## Retrieval misses and ranking differences", ""]
    differences = []
    for b, h in zip(result["outcomes"]["baseline"], result["outcomes"]["hybrid"], strict=True):
        if b["answerable"] and (
            b["evidenceRecall"] < 1 or h["evidenceRecall"] < 1 or b["evidenceRanks"] != h["evidenceRanks"]
        ):
            differences.append((b, h))
    if differences:
        lines += [
            "Evidence ranks are per labeled target; `missing` means the required quote was absent.",
            "",
            "| ID | Question | Baseline evidence ranks | Hybrid evidence ranks |",
            "| --- | --- | --- | --- |",
        ]
        for b, h in differences:

            def ranks(row):
                return ", ".join(str(rank) if rank else "missing" for rank in row["evidenceRanks"])

            lines.append(f"| {b['id']} | {cell(b['question'])} | {ranks(b)} | {ranks(h)} |")
    else:
        lines.append("Both modes retrieved all labeled evidence at identical ranks on the answerable cases.")
    lines += [
        "",
        "## Unsupported questions",
        "",
        "A high similarity score can occur for an unsupported question that resembles a policy. "
        "Passing the threshold means generation would be allowed, not that a correct answer exists.",
        "",
        "| ID | Question | Baseline score / threshold action | Hybrid score / threshold action |",
        "| --- | --- | --- | --- |",
    ]
    for b, h in zip(result["outcomes"]["baseline"], result["outcomes"]["hybrid"], strict=True):
        if not b["answerable"]:

            def decision(row):
                return f"{row['confidence']:.3f} / {'block' if row['wouldAbstain'] else 'allow generation'}"

            lines.append(f"| {b['id']} | {cell(b['question'])} | {decision(b)} | {decision(h)} |")
    lines += [
        "",
        "## Setup and environment",
        "",
        f"- Platform: {machine['platform']}; Python {machine['python']}.",
        f"- CPU: {machine['cpu']}; {machine['logicalCpuCount']} logical CPUs; "
        f"{machine.get('physicalRamGiB', 'unmeasured')} GiB physical RAM; "
        f"PyTorch intra-operation threads: {config['torchThreads']}.",
        f"- Cached-model initialization and one probe: {result['setup']['modelInitializationMs'] / 1000:.2f} seconds.",
        f"- Parsing, document embedding, and storage: {result['setup']['ingestionMs'] / 1000:.2f} seconds "
        "with the model already loaded.",
        f"- Index-cold / model-warm first retrieval: baseline "
        f"{result['setup']['indexColdModelWarmMs']['baseline']:.2f} ms; hybrid "
        f"{result['setup']['indexColdModelWarmMs']['hybrid']:.2f} ms.",
        f"- Dataset SHA-256: `{result['dataset']['sha256']}`.",
        f"- Canonical parsed chunks and embedding hashes SHA-256: `{corpus['chunksSha256']}`.",
        f"- Cached model refs: `{json.dumps(config['modelRefs'], sort_keys=True)}`.",
        "",
        "| Package | Version |",
        "| --- | --- |",
    ]
    lines += [f"| {name} | {version} |" for name, version in machine["packages"].items()]
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "- Use the observed evidence recall and latency together. A hybrid pipeline performs "
        "extra embedding/search work and is not expected to be a universal latency improvement.",
        "- Inspect the per-question failures and category breakdown before claiming hybrid is "
        "better. Retrieving the right document is weaker evidence than retrieving the answer passage.",
        "- This is a small, synthetic development set authored with the corpus, without independent "
        "annotators, a held-out test set, or statistical significance analysis. Do not generalize "
        "these percentages to real users, other languages, or larger corpora.",
        "- All inputs are Markdown and use the same parser/chunker as uploads. This does not "
        "benchmark PDF/Office parsing quality or scanned-document OCR.",
        "- Quote labels identify expected evidence, not every potentially relevant passage. "
        "Version and multi-source cases test retrieval, not whether a model interprets dates or resolves conflicts.",
        "- No ranking weights, chunking parameters, or confidence threshold were tuned after observing results.",
        "- Latency is from one local process, without a controlled idle machine, concurrency load, "
        "confidence intervals, or production network conditions. Memory, disk, and ChromaDB "
        "performance comparisons are outside this run.",
        "",
        "## Reproduce",
        "",
        "From the repository root, with the existing dependencies and cached MiniLM model:",
        "",
        "```powershell",
        f".\\.venv\\Scripts\\python.exe -m scripts.evaluate_retrieval --repeats {config['repeats']} --top-k {k}",
        "```",
        "",
        "No virtual environment activation, running API/frontend, or NVIDIA key is required. "
        "The script runs offline by default and fails if real embeddings are unavailable; "
        "it never substitutes stub vectors. On a new machine, add `--allow-model-download` "
        "for the initial Hugging Face download.",
        "",
        "Artifacts: [dataset and labels](evaluation/dataset.json), "
        "[raw results, environment, source hashes, retrieved snippets, and timing samples](evaluation/results.json), "
        "[runner](../scripts/evaluate_retrieval.py). Re-running replaces this report and results.json.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "docs/evaluation/dataset.json")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/EVALUATION_REPORT.md")
    parser.add_argument("--results", type=Path, default=ROOT / "docs/evaluation/results.json")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100 or not 1 <= args.top_k <= 20:
        parser.error("repeats must be 1..100 and top-k must be 1..20")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    result = run_benchmark(
        dataset, repeats=args.repeats, top_k=args.top_k, allow_download=args.allow_model_download
    )
    for path in (args.report, args.results):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.report.write_text(render_report(result), encoding="utf-8")
    print(f"Report written: {args.report}", flush=True)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
