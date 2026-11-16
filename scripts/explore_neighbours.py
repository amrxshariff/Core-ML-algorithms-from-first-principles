# making synthetic dataset for testing

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from neighbours.distances import cosine_distance, euclidean_distance, manhattan_distance


def as_pairwise(distance_func):
    """Adapt a vectorised (a, B) distance function to sklearn's callable-metric signature.

    sklearn calls a callable metric with two 1-D points, whereas our functions take one
    point and a 2-D matrix of points, so wrap b as a single-row matrix and unwrap the result.
    """
    return lambda a, b: distance_func(a, b.reshape(1, -1))[0]


def main():
    X, y = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=1.0, random_state=42)

    print(f"Shape of feature matrix X: {X.shape}")
    print(f"Shape of target vector y: {y.shape}")
    print(f"Unique classes in y: {np.unique(y)}")

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('2D Synthetic Dataset')
    plt.show()

    # splitting into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Training a simple KNN classifier using the custom distance metrics
    for name, distance_func in [
        ("euclidean", euclidean_distance),
        ("manhattan", manhattan_distance),
        ("cosine", cosine_distance),
    ]:
        classifier = KNeighborsClassifier(n_neighbors=5, metric=as_pairwise(distance_func))
        classifier.fit(X_train, y_train)
        accuracy = np.mean(classifier.predict(X_test) == y_test)
        print(f"{name} test accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
