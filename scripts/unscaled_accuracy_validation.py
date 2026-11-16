import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from neighbours.knn_scratch import KNN as customKNN


def unscaled_accuracy_validation():
    # 1. Generate data and split
    X, y = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=2.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # 2. Scale feature 0 by 100 WITHOUT standardizing (no StandardScaler)
    X_train_distorted = X_train.copy()
    X_test_distorted = X_test.copy()
    
    X_train_distorted[:, 0] *= 100
    X_test_distorted[:, 0] *= 100

    # 3. Evaluate accuracy across k
    accuracy = []
    k_values = list(range(1, 26))

    for k in k_values:
        custom_knn = customKNN(n_neighbours=k, distance_metric='euclidean')
        custom_knn.fit(X_train_distorted, y_train)
        
        predictions = custom_knn.predict(X_test_distorted)
        acc = np.mean(predictions == y_test)
        accuracy.append(acc)

    # 4. Plot the distorted validation curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracy, marker='o', linestyle='-', color='crimson', label='Unscaled Feature (*100) Accuracy')
    
    plt.axvline(x=3, color='black', linestyle='--', linewidth=1.5, label='Overfitting (Low k)')
    plt.axvline(x=23, color='black', linestyle='--', linewidth=1.5, label='Underfitting (High k)')
    
    plt.title("k-NN Accuracy vs. k (Unscaled Feature Dominated)")
    plt.xlabel("Hyperparameter k")
    plt.ylabel("Classification Accuracy")
    plt.xticks(k_values)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    unscaled_accuracy_validation()