from time import perf_counter

import numpy as np
from sklearn.datasets import make_blobs

from neighbours.distances import euclidean_distance


def time_benchmarking(query, training):
    """Time a pure-Python loop computing the Euclidean distance from query to each training point"""
    start_time = perf_counter()
    history = []
    for point in training:
        distance = np.sqrt(np.sum((point-query)**2))
        history.append(distance)
    elapsed = perf_counter() - start_time

    return elapsed


def custom_model(query, training):
    """Time the vectorised euclidean_distance over the same training points"""
    start_time = perf_counter()
    euclidean_distance(query, training)
    elapsed = perf_counter() - start_time

    return elapsed


if __name__ == "__main__":
    X, y = make_blobs(n_samples=5000, n_features=2, centers=3, cluster_std=1.0, random_state=42)

    query = X[0]
    training = X[1:]

    manual = time_benchmarking(query, training)
    custom_mod = custom_model(query, training)
    first = (np.sum(custom_mod/custom_mod))
    second = (np.sum(manual/custom_mod))
    print(f"ratio = [{first}]:[{second}]")
