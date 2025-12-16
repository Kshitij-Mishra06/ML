# -------------------------------
# Mall Customers Regression Model
# -------------------------------

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Step 2: Load the dataset
data = pd.read_csv("Mall_Customers.csv")

# Step 3: Explore dataset
print("Dataset Shape:", data.shape)
print(data.head())

# Step 4: Check for missing values
print("\nMissing Values:\n", data.isnull().sum())

# Step 5: Encode categorical column (Genre)
le = LabelEncoder()
data['Genre'] = le.fit_transform(data['Genre'])  # Male=1, Female=0

# Step 6: Define features and target
X = data[['Genre', 'Age', 'Annual Income (k$)']]
y = data['Spending Score (1-100)']

# Step 7: Split data into train-test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 8: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 9: Make predictions
y_pred = model.predict(X_test)

# Step 10: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", round(mse, 2))
print("R² Score:", round(r2, 2))

# Step 11: Compare Actual vs Predicted
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
print("\nSample Predictions:\n", results.head())

# Step 12: Visualization
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k')
plt.xlabel("Actual Spending Score")
plt.ylabel("Predicted Spending Score")
plt.title("Actual vs Predicted Spending Score")
plt.grid(True)
plt.show()

# Step 13: Model Coefficients
print("\nModel Coefficients:")
for col, coef in zip(X.columns, model.coef_):
    print(f"{col}: {coef:.2f}")
print("Intercept:", model.intercept_)
