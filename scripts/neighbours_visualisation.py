import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

from neighbours.knn_scratch import KNN as customKNN


def visualisation_neighbours():
    """
    Decision boundary under Euclidean distance drawn on a mesh grid

    """
    X_train, Y_train = make_blobs(
        n_samples=300, n_features=2, centers=3, cluster_std=1.0, random_state=42
    )

    x0 = np.linspace(-11, 11)
    x1 = np.linspace(-11, 11)
    xx, yy = np.meshgrid(x0, x1)

    flatten = xx.flatten()
    flatten1 = yy.flatten()

    combine = np.column_stack((flatten, flatten1))

    metrics = ['euclidean', 'manhattan', 'cosine']

    _, axes = plt.subplots(1,3, figsize = (18,6), sharex = True, sharey = True)

    for ax, metric in zip(axes, metrics):
        custom_knn = customKNN(n_neighbours=5, distance_metric = metric)
        custom_knn.fit(X_train, Y_train)
        
        predictions = custom_knn.predict(combine)
        
        reshape = predictions.reshape(xx.shape)
        
        ax.contourf(xx, yy, reshape, alpha = 0.4, cmap = plt.cm.Spectral)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, edgecolors='k', cmap=plt.cm.Spectral)
        ax.set_title(f"k-NN Decision Boundary ({metric.capitalize()})")
        ax.set_xlabel("feature 0")

    axes[0].set_ylabel("Feature 1")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualisation_neighbours()
