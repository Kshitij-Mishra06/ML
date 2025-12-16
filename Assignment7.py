import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Preprocess: Map Genre to numerical labels (Male=0, Female=1)
data['Genre'] = data['Genre'].map({'Male': 0, 'Female': 1})

# Features: Age, Annual Income, Spending Score
features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
labels = data['Genre'].values

# Add bias term (column of ones)
X = np.c_[np.ones(features.shape[0]), features]
y = labels

# Split into train and test sets (80-20 split)
np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.8 * len(X))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]

# Single Perceptron implementation
class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, z):
        return 1 if z >= 0 else 0

    def predict(self, x):
        z = np.dot(self.weights, x)
        return self.activation(z)

    def fit(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                update = self.learning_rate * (target - prediction)
                self.weights += update * xi

# Train the perceptron
perceptron = Perceptron(input_size=X_train.shape[1] - 1)  # -1 because X includes bias
perceptron.fit(X_train, y_train)

# Evaluate on test set
correct = 0
for xi, target in zip(X_test, y_test):
    prediction = perceptron.predict(xi)
    if prediction == target:
        correct += 1
accuracy = correct / len(X_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Example prediction
test_point = np.array([1, 30, 50, 60])  # Bias=1, Age=30, Income=50, Spending=60
prediction = perceptron.predict(test_point)
genre = 'Female' if prediction == 1 else 'Male'
print(f"Predicted genre for Age=30, Income=50k, Spending=60: {genre}")
