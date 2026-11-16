import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from neighbours.knn_scratch import KNN as CustomKNN


@pytest.mark.parametrize("distance_metric", ["euclidean", "manhattan", "cosine"])
def test_k_equals_one_is_exact(distance_metric):
    """Generate synthetic dataset"""
    X, y = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=1.0, random_state=42)


    sc = StandardScaler()
    X_train = sc.fit_transform(X)
    

    # Train custom KNN
    custom_knn = CustomKNN(n_neighbours=1, distance_metric=distance_metric)
    custom_knn.fit(X_train, y)

    # Make predictions
    custom_pred = custom_knn.predict(X_train)

    np.testing.assert_array_equal(custom_pred, y)

if __name__ == "__main__":
    test_k_equals_one_is_exact("euclidean")
    test_k_equals_one_is_exact("manhattan")
    test_k_equals_one_is_exact("cosine")
    print("All tests passed.")