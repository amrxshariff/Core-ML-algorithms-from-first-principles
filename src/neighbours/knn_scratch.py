import numpy as np

from neighbours.distances import cosine_distance, euclidean_distance, manhattan_distance


class KNN:
    def __init__(self, n_neighbours, distance_metric):
        self.n_neighbours = n_neighbours

        self.distance_metric = {'euclidean': euclidean_distance,
                                'manhattan': manhattan_distance,
                                'cosine':cosine_distance}

        if distance_metric not in self.distance_metric:
            raise ValueError(f"Invalid distance metric. Choose from {list(self.distance_metric.keys())}")
        self.distance_func = self.distance_metric[distance_metric]

    def fit(self, X, y):
        """Storing training data matrices"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        """Predict classes for query points in X, tie breaking rule is that argmax naturally returns the first index it comes across in case of ties"""
        X = np.array(X)
        predictions = []

        for x in X:
            distances = self.distance_func(x, self.X_train)
            nearest_indices = np.argpartition(distances, self.n_neighbours)[:self.n_neighbours]
            nearest_labels = self.y_train[nearest_indices]
            predictions.append(np.bincount(nearest_labels).argmax())

        return np.array(predictions)





