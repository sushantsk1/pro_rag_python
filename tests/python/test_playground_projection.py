import json

import numpy as np
import pytest

from server.playground_projection import project_embeddings


def project(matrix, query):
    return project_embeddings([{"id": f"C{i}"} for i in range(len(matrix))], matrix, query)


def test_projection_matches_svd_geometry_and_full_cosine():
    matrix = np.random.default_rng(42).normal(size=(12, 24))
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    query = matrix[0]
    result = project(matrix, query)
    coordinates = np.array([p["coordinates"] for p in result["points"]])
    centered = matrix - matrix.mean(axis=0)
    _, singular, basis = np.linalg.svd(centered, full_matrices=False)
    expected = centered @ basis[:3].T
    assert coordinates @ coordinates.T == pytest.approx(expected @ expected.T)
    assert result["explainedVariance"] == pytest.approx((singular[:3] ** 2) / (singular**2).sum())
    assert result["query"]["coordinates"] == pytest.approx(coordinates[0])
    assert result["points"][0]["cosineSimilarity"] == pytest.approx(1)
    assert [p["cosineSimilarity"] for p in result["points"]] == pytest.approx(matrix @ query)


def test_question_does_not_move_chunks():
    matrix = np.random.default_rng(10).normal(size=(8, 20))
    first = project(matrix, matrix[0])
    second = project(matrix, matrix[-1])
    assert [p["coordinates"] for p in first["points"]] == [p["coordinates"] for p in second["points"]]
    assert first["query"]["coordinates"] != second["query"]["coordinates"]


@pytest.mark.parametrize("matrix", [[[1, 0, 0]], [[1, 0], [1, 0]], [[0, 0], [0, 0]], [[1], [-1]]])
def test_degenerate_and_small_maps_are_finite(matrix):
    result = project(matrix, matrix[0])
    json.dumps(result, allow_nan=False)
    assert all(len(p["coordinates"]) == 3 for p in result["points"])
    assert result["componentsAvailable"] <= len(matrix) - 1


def test_out_of_vocabulary_query_has_no_cosine_or_search_marker():
    result = project([[1, 0], [0, 1]], [0, 0])
    assert result["query"]["hasSignal"] is False
    assert all(p["cosineSimilarity"] is None for p in result["points"])


def test_api_returns_projection_for_actual_index_vectors(client):
    run = client.post("/api/v1/playground/run", json={"embedding": "tfidf", "untilStage": 2}).json()
    output = run["stages"][2]["output"]
    projection = output["projection"]
    assert [p["chunkId"] for p in projection["points"]] == [v["chunkId"] for v in output["vectors"]]
    for point, vector in zip(projection["points"], output["vectors"], strict=True):
        assert point["cosineSimilarity"] == pytest.approx(
            np.dot(vector["values"], output["queryVector"]), abs=1e-6
        )
    assert all(stage["status"] == "not_run" for stage in run["stages"][3:])
