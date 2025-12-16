import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Select features for clustering: Age, Annual Income, Spending Score
features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

# K-means implementation
def kmeans(X, k, max_iters=100):
    # Initialize centroids randomly
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        # Assign each point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    return centroids, labels

# Perform K-means with k=5 (common for this dataset)
k = 5
centroids, labels = kmeans(features, k)

# Print centroids
print("Centroids:")
for i, centroid in enumerate(centroids):
    print(f"Cluster {i+1}: Age={centroid[0]:.2f}, Income={centroid[1]:.2f}, Spending={centroid[2]:.2f}")

# Calculate inertia (sum of squared distances)
def calculate_inertia(X, centroids, labels):
    inertia = 0
    for i in range(len(X)):
        centroid = centroids[labels[i]]
        inertia += np.sum((X[i] - centroid) ** 2)
    return inertia

inertia = calculate_inertia(features, centroids, labels)
print(f"Inertia: {inertia:.2f}")

# Visualize clusters (2D plot using Income and Spending)
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'orange', 'purple']
for i in range(k):
    cluster_points = features[labels == i]
    plt.scatter(cluster_points[:, 1], cluster_points[:, 2], c=colors[i], label=f'Cluster {i+1}')
plt.scatter(centroids[:, 1], centroids[:, 2], c='black', marker='x', s=100, label='Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('K-means Clustering (k=5)')
plt.legend()
plt.savefig('kmeans_clusters.png')
print("Plot saved as 'kmeans_clusters.png'")
