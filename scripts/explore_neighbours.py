# making synthetic dataset for testing

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

X, y = make_blobs(n_samples = 300, n_features = 2, centers = 3, cluster_std = 1.0, random_state = 42)

print(f"Shape of feature matrix X: {X.shape}")
print(f"Shape of target vector y: {y.shape}")
print(f"Unique classes in y: {np.unique(y)}")

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('2D Synthetic Dataset')
plt.show()

#splitting into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

# Euclidean / Manhattan / Cosine

# B takes each row as a point, a is a single point. Returns distance of a from each point in B (training set)

def euclidean_distance(a, B):
    distance = np.sqrt(np.sum((a-B)**2, axis=1))
    return distance

def manhattan_distance(a, B):
    distance = np.sum(np.abs(a-B), axis=1)
    return distance

def cosine_distance(a, B):
    dot_product = np.dot(B, a)
    norm_a = np.linalg.norm(a)
    norm_B = np.linalg.norm(B, axis=1)
    cosine_similarity = dot_product / ((norm_a * norm_B)+1e-10)  # Adding a small epsilon to avoid division by zero
    distance = 1 - cosine_similarity
    return distance


# Vectorisation already done in the above functions. We can also use sklearn's pairwise_distances function for this.

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training a simple KNN classifier using the above distance metrics

classifier_euclidean = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
classifier_euclidean.fit(X_train, y_train)

classifier_manhattan = KNeighborsClassifier(n_neighbors = 5, metric = 'manhattan')
classifier_manhattan.fit(X_train, y_train)

classifier_cosine = KNeighborsClassifier(n_neighbors = 5, metric = 'cosine')
classifier_cosine.fit(X_train,y_train)

print(classifier_euclidean.predict(sc.transform([[0.5, 0.5]])))
print(classifier_manhattan.predict(sc.transform([[0.5, 0.5]])))
print(classifier_cosine.predict(sc.transform([[0.5, 0.5]])))    
