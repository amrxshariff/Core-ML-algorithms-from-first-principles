import numpy as np


def euclidean_distance(a, B):
    distance = np.sqrt(np.sum((a-B)**2, axis=1))
    return distance

def manhattan_distance(a, B):
    distance = np.sum(np.abs(a-B), axis=1)
    return distance

def cosine_distance(a, B):
    return 1 - (B @ a) / (np.linalg.norm(B, axis=1) * np.linalg.norm(a))