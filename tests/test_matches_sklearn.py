import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as sklearnKNN
from sklearn.preprocessing import StandardScaler

from neighbours.knn_scratch import KNN as CustomKNN


@pytest.mark.parametrize("distance_metric", ["euclidean", "manhattan", "cosine"])
def test_matches_sklearn(distance_metric):
    """Generate synthetic dataset"""
    X, y = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=1.0, random_state=42)

    # Split into train and test sets
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size = 0.25, random_state=42)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Train sklearn KNN
    sklearn_knn = sklearnKNN(n_neighbors=5, metric=distance_metric)
    sklearn_knn.fit(X_train, y_train)

    # Train custom KNN
    custom_knn = CustomKNN(n_neighbours=5, distance_metric=distance_metric)
    custom_knn.fit(X_train, y_train)

    # Make predictions
    sklearn_pred = sklearn_knn.predict(X_test)
    custom_pred = custom_knn.predict(X_test)

    # Assert predictions are equal
    np.testing.assert_array_equal(sklearn_pred, custom_pred)

if __name__ == "__main__":
    test_matches_sklearn("euclidean")
    test_matches_sklearn("manhattan")
    test_matches_sklearn("cosine")
    print("All tests passed.")