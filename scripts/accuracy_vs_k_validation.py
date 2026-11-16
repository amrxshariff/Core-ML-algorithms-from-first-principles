import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neighbours.knn_scratch import KNN as customKNN


def accuracy_k_validation():
    # increased cluster_std to ensure that points weren't so close to centroids in clusters
    X, y = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=5.0, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    accuracy = []
    
    for k in range (1,26):
       
        custom_knn = customKNN(n_neighbours=k,distance_metric = 'euclidean')
        custom_knn.fit(X_train_scaled, y_train)
                        
        predictions = custom_knn.predict(X_test_scaled)
        acc = np.mean(predictions == y_test)
        accuracy.append(acc)

    k_values = list(range(1,26))
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracy, marker='o', linestyle='-', color='b', label='Test Accuracy')
    plt.axvline(x=3, color='red', linestyle='--', linewidth=2, label='Overfitting (Low k)')
    plt.axvline(x=23, color='red', linestyle='--', linewidth=2, label='Underfitting (High k)')
    plt.title("k-NN Accuracy vs. k (Validation Curve)")
    plt.xlabel("Hyperparameter k")
    plt.ylabel("Classification Accuracy")
    plt.xticks(k_values)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    accuracy_k_validation()


   

# tie breaker case not implemented which may result in lower performance for even k values        
# overfitting at a low k - accuracy didn't peak once
# underfitting at a high k gave higher accuracy results but only one result that matched the peak accuracy
# sweet spot approximately 0.893    
# sharp drops immediately after peaks which could highlight the sensitivity of K-NN to local descision boundary shifts when adding a single neighbour to the voting pool                 
       
         
        