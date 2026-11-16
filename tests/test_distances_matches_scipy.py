import numpy as np
import pytest
from scipy.spatial.distance import cdist

from neighbours.distances import cosine_distance, euclidean_distance, manhattan_distance


@pytest.mark.parametrize("metric_name, custom_func, scipy_metric", [
    ("euclidean", euclidean_distance, "euclidean"),
    ("manhattan", manhattan_distance, "cityblock"),
    ("cosine", cosine_distance, "cosine"),])
def test_distances_match_scipy(metric_name, custom_func, scipy_metric):
    """Custom distances from one query point to every training point match scipy's cdist"""
    rng = np.random.default_rng(seed=42)
    X_query = rng.random((1, 5))  # 1 query point
    X_train = rng.random((10, 5))  # 10 training points

    custom_distances = custom_func(X_query[0], X_train)

    scipy_distances = cdist(X_query, X_train, metric=scipy_metric)[0]

    np.testing.assert_allclose(custom_distances, scipy_distances, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    test_distances_match_scipy("euclidean", euclidean_distance, "euclidean")
    test_distances_match_scipy("manhattan", manhattan_distance, "cityblock")
    test_distances_match_scipy("cosine", cosine_distance, "cosine")
    print("All tests passed.")
