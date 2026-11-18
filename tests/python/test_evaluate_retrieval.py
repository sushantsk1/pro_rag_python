import json

import pytest

from scripts.evaluate_retrieval import ROOT, percentile, score_case, summarize, validate_dataset


def result(sources, confidence=0.5):
    return {
        "sources": [
            {"documentName": name, "snippet": text, "location": {}, "scores": {}} for name, text in sources
        ],
        "confidence": confidence,
    }


def test_correct_document_wrong_passage_is_not_evidence():
    case = {
        "id": "a",
        "category": "direct",
        "question": "How long?",
        "evidence": [{"document": "policy.md", "quote": "24 months"}],
    }
    row = score_case(case, result([("policy.md", "Accidental damage is excluded.")]))
    assert row["documentHit"] is True
    assert row["documentRecall"] == 1
    assert row["documentReciprocalRank"] == 1
    assert row["evidenceRecall"] == 0
    assert row["allEvidenceFound"] is False
    assert row["evidenceRanks"] == [None]


def test_multi_source_partial_credit_and_rank():
    case = {
        "id": "a",
        "category": "multi-source",
        "question": "Both?",
        "evidence": [
            {"document": "current.md", "quote": "24 months"},
            {"document": "legacy.md", "quote": "12 months"},
        ],
    }
    row = score_case(case, result([("other.md", "Unrelated"), ("legacy.md", "Lasts 12 MONTHS.")]))
    assert row["documentHit"] is True
    assert row["documentRecall"] == 0.5
    assert row["evidenceRecall"] == 0.5
    assert row["allEvidenceFound"] is False
    assert row["evidenceReciprocalRank"] == 0.5
    assert row["evidenceRanks"] == [None, 2]


def test_unanswerable_cases_excluded_from_relevance_and_threshold_boundary():
    positive = {
        "id": "a",
        "category": "direct",
        "question": "Covered?",
        "evidence": [{"document": "a.md", "quote": "covered"}],
    }
    negative = {"id": "b", "category": "unanswerable", "question": "Unknown?", "evidence": []}
    rows = [
        score_case(positive, result([("a.md", "covered")], 0.3)),
        score_case(negative, result([], 0.299)),
        score_case(negative, result([], 0.8)),
    ]
    summary = summarize(rows, [{"durationMs": n} for n in range(1, 21)])
    assert summary["documentHitRate"] == 1
    assert summary["evidenceRecall"] == 1
    assert summary["documentMRR"] == 1
    assert summary["unsupportedAbstentionRate"] == 0.5
    assert summary["answerableFalseAbstentionRate"] == 0
    assert summary["thresholdDecisionAccuracy"] == pytest.approx(2 / 3)
    assert summary["warmMedianMs"] == 10.5
    assert summary["warmP95Ms"] == 19


def test_filter_leakage_counted_even_when_expected_evidence_found():
    case = {
        "id": "a",
        "category": "filtered",
        "question": "How long?",
        "filterDocuments": ["a.md"],
        "evidence": [{"document": "a.md", "quote": "24 months"}],
    }
    row = score_case(case, result([("a.md", "24 months"), ("b.md", "12 months")]))
    assert row["evidenceRecall"] == 1
    assert row["filterViolationCount"] == 1


def test_committed_dataset_has_valid_evidence_labels():
    dataset = json.loads((ROOT / "docs/evaluation/dataset.json").read_text(encoding="utf-8"))
    validate_dataset(dataset)
    dataset["cases"][0]["evidence"][0]["quote"] = "a nonexistent passage"
    with pytest.raises(ValueError, match="Missing evidence quote"):
        validate_dataset(dataset)


def test_nearest_rank_percentile_small_sample():
    assert percentile([3, 1, 2], 0.95) == 3
