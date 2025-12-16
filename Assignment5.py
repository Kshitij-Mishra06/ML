import numpy as np
import pandas as pd
from collections import Counter
import math

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Preprocess: Map Genre to numerical labels
data['Genre'] = data['Genre'].map({'Male': 0, 'Female': 1})

# Features: Age, Annual Income, Spending Score
features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
labels = data['Genre'].values

# Split into train and test sets (80-20 split)
np.random.seed(42)
indices = np.random.permutation(len(data))
train_size = int(0.8 * len(data))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = features[train_indices]
y_train = labels[train_indices]
X_test = features[test_indices]
y_test = labels[test_indices]

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

# KNN implementation
def knn_predict(X_train, y_train, test_point, k):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], test_point)
        distances.append((dist, y_train[i]))
    # Sort by distance
    distances.sort(key=lambda x: x[0])
    # Get k nearest neighbors
    k_neighbors = distances[:k]
    # Majority vote
    k_labels = [label for _, label in k_neighbors]
    most_common = Counter(k_labels).most_common(1)[0][0]
    return most_common

# Evaluate on test set
def evaluate_knn(X_train, y_train, X_test, y_test, k):
    correct = 0
    for i in range(len(X_test)):
        prediction = knn_predict(X_train, y_train, X_test[i], k)
        if prediction == y_test[i]:
            correct += 1
    accuracy = correct / len(X_test)
    return accuracy

# Test with k=5
k = 5
accuracy = evaluate_knn(X_train, y_train, X_test, y_test, k)
print(f"Accuracy with k={k}: {accuracy:.2f}")

# Example prediction
test_point = [30, 50, 60]  # Example: Age=30, Income=50k, Spending=60
prediction = knn_predict(X_train, y_train, test_point, k)
genre = 'Female' if prediction == 1 else 'Male'
print(f"Predicted genre for Age=30, Income=50k, Spending=60: {genre}")
