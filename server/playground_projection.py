"""Small, deterministic PCA views of experiment vectors, independent of retrieval."""

import numpy as np


def project_embeddings(chunks, vectors, query):
    # Fit on chunks only: changing the question must not rearrange the document map.
    matrix = np.asarray(vectors, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0)
    query_centered = query - matrix.mean(axis=0)
    # The Gram matrix is at most 256x256, including for wide TF-IDF vocabularies.
    eigenvalues, eigenvectors = np.linalg.eigh(centered @ centered.T)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    eigenvectors = eigenvectors[:, order]
    tolerance = max(float(eigenvalues[0]), 1) * 1e-10
    rank = min(3, int(np.count_nonzero(eigenvalues > tolerance)))
    coordinates = np.zeros((len(chunks), 3))
    query_coordinates = np.zeros(3)
    explained = np.zeros(3)
    for component in range(rank):
        axis = eigenvectors[:, component]
        # Resolve the arbitrary eigenvector sign without depending on the question.
        sign = 1 if axis[np.argmax(np.abs(axis))] >= 0 else -1
        axis = axis * sign
        scale = np.sqrt(eigenvalues[component])
        coordinates[:, component] = axis * scale
        query_coordinates[component] = query_centered @ (centered.T @ axis / scale)
        explained[component] = eigenvalues[component] / eigenvalues.sum()
    query_norm = float(np.linalg.norm(query))
    norms = np.linalg.norm(matrix, axis=1)
    similarities = [
        float(np.clip(np.dot(row, query) / (norm * query_norm), -1, 1))
        if norm > 0 and query_norm > 0
        else None
        for row, norm in zip(matrix, norms, strict=True)
    ]
    return {
        "method": "PCA",
        "fit": "chunks",
        "componentsAvailable": rank,
        "explainedVariance": explained.tolist(),
        "points": [
            {"chunkId": chunk["id"], "coordinates": point.tolist(), "cosineSimilarity": similarity}
            for chunk, point, similarity in zip(chunks, coordinates, similarities, strict=True)
        ],
        "query": {"coordinates": query_coordinates.tolist(), "hasSignal": query_norm > 0},
    }
