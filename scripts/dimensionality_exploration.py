import matplotlib.pyplot as plt
import numpy as np


def generate_uniform_points(n,d):
    """Generates n uniform random points in d dimensions in the form of an array of shape (n,d)"""
    return np.random.rand(n,d)

def nearest_neighbour_ratio():
    d_values = np.unique(np.logspace(np.log10(2), np.log10(1000), num = 40, dtype = int))

    n_database = 500 #reference space
    n_queries = 100 #query points to measure distances against database
    mean_ratios = []
    threshold_d = None

    for d in d_values:
        db_points = generate_uniform_points(n_database, d)
        query_points = generate_uniform_points(n_queries, d)
        
        # Vectorized pairwise Euclidean distance calculation: shape (n_queries, n_database)
        distances = np.linalg.norm(query_points[:, np.newaxis, :] - db_points[np.newaxis, :, :], axis=2)
        
        # Find nearest and farthest neighbor distances for each query point
        min_dist = np.min(distances, axis=1)
        max_dist = np.max(distances, axis=1)
        
        # Average ratio of nearest to farthest neighbor distance
        avg_ratio = np.mean(min_dist / max_dist)
        mean_ratios.append(avg_ratio)
        
        # Record the first dimension where the ratio crosses 0.9
        if avg_ratio >= 0.9 and threshold_d is None:
            threshold_d = d
            print(f"Dimension where ratio passes 0.9: d = {threshold_d}")

    # Plotting the results on a log x-axis
    plt.figure(figsize=(10, 6))
    plt.plot(d_values, mean_ratios, marker='o', linestyle='-', color='purple', label='Mean (Nearest / Farthest) Ratio')
    if threshold_d:
        plt.axvline(x=threshold_d, color='red', linestyle='--', label=f'Threshold 0.9 (d={threshold_d})')
    
    plt.xscale('log')
    plt.title("Curse of Dimensionality: Nearest vs. Farthest Neighbor Distance Ratio")
    plt.xlabel("Dimensionality d (Log Scale)")
    plt.ylabel("Ratio (Nearest / Farthest)")
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    nearest_neighbour_ratio()

# As the dimensionality scales up on the logarithmic axis, 
# ratio smoothly climbs to 1 which suggests that the distance to a point's nearest neighbour becomes 
# indistinguishable from the distance to it's farthest neighbour

# Takeaway, K-NN applied to higher dimensions makes the predictions obsolete and forces the model to guess a particular class rather than consider the votes for each class
# When the ratio of nearest to farthest neighbours converges to 1, death of locality occurs - 
# when every point in the dataset is roughly equidistant from query point, neighbourhood is no longer local and k nearest neighbours 
# are no longer closer to the query than random selection of points across the entire dataset, and degenerate voting - 
# since neighbours are effectively randomised due to distance concentration, majority vote no longer has it's discriminative power and so the model stops learning a decision boundary and
# resorts to reflecting the global class distribution (guessing) which plummets accuracy