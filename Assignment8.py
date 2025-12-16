import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Preprocess: Map Genre to numerical labels (Male=0, Female=1)
data['Genre'] = data['Genre'].map({'Male': 0, 'Female': 1})

# Features: Age, Annual Income, Spending Score
features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
labels = data['Genre'].values

# Normalize features
features = (features - features.mean(axis=0)) / features.std(axis=0)

# Split into train and test sets (80-20 split)
np.random.seed(42)
indices = np.random.permutation(len(features))
train_size = int(0.8 * len(features))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = features[train_indices]
y_train = labels[train_indices].reshape(-1, 1)
X_test = features[test_indices]
y_test = labels[test_indices].reshape(-1, 1)

# Neural Network with Backpropagation
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, epochs=1000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Initialize weights
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        # Output layer error
        d_output = (output - y) * self.sigmoid_derivative(output)
        
        # Hidden layer error
        d_hidden = np.dot(d_output, self.W2.T) * self.sigmoid_derivative(self.a1)
        
        # Gradients
        dW2 = np.dot(self.a1.T, d_output)
        db2 = np.sum(d_output, axis=0, keepdims=True)
        dW1 = np.dot(X.T, d_hidden)
        db1 = np.sum(d_hidden, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X, y):
        for epoch in range(self.epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 100 == 0:
                loss = np.mean((output - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int)

# Train the network
nn = NeuralNetwork(input_size=3, hidden_size=4, output_size=1)
nn.train(X_train, y_train)

# Evaluate on test set
predictions = nn.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Example prediction
test_point = np.array([[30, 50, 60]])
test_point = (test_point - features.mean(axis=0)) / features.std(axis=0)  # Normalize
prediction = nn.predict(test_point)[0][0]
genre = 'Female' if prediction == 1 else 'Male'
print(f"Predicted genre for Age=30, Income=50k, Spending=60: {genre}")
