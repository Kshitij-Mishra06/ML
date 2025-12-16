# -------------------------------
# SVM Classification on Mall Customers Dataset
# Converting Spending Score to Classes: Low (1-33), Medium (34-66), High (67-100)
# -------------------------------

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Step 2: Load the Mall Customers dataset
data = pd.read_csv("Mall_Customers.csv")

# Step 3: Explore dataset
print("Dataset Shape:", data.shape)
print(data.head())

# Step 4: Check for missing values
print("\nMissing Values:\n", data.isnull().sum())

# Step 5: Encode categorical column (Genre)
le = LabelEncoder()
data['Genre'] = le.fit_transform(data['Genre'])  # Male=1, Female=0

# Step 6: Bin Spending Score into classes
def bin_spending_score(score):
    if score <= 33:
        return 0  # Low
    elif score <= 66:
        return 1  # Medium
    else:
        return 2  # High

data['Spending_Class'] = data['Spending Score (1-100)'].apply(bin_spending_score)

# Step 7: Define features and target
X = data[['Genre', 'Age', 'Annual Income (k$)']]
y = data['Spending_Class']

# Step 8: Preprocess data (Standardize features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 9: Split data into train-test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Step 10: Train SVM model (using RBF kernel)
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
model.fit(X_train, y_train)

# Step 11: Make predictions
y_pred = model.predict(X_test)

# Step 12: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Evaluation:")
print("Accuracy:", round(accuracy, 2))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 13: Visualization - Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Step 14: Visualize decision boundaries using PCA (reduce to 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y, test_size=0.3, random_state=42
)

model_pca = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
model_pca.fit(X_train_pca, y_train_pca)

# Create a mesh to plot decision boundaries
h = .02  # step size in the mesh
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = model_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_pca, edgecolors='k', marker='o', label='Train')
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test_pca, edgecolors='k', marker='s', label='Test')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('SVM Decision Boundaries (PCA)')
plt.legend()
plt.show()
