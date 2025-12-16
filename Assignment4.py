import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Preprocess the data
# Encode Genre: Male -> 0, Female -> 1
le = LabelEncoder()
data['Genre'] = le.fit_transform(data['Genre'])

# Features and target
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
y = data['Genre']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Bagging: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Bagging (Random Forest) Accuracy: {rf_accuracy:.2f}")

# Boosting: AdaBoost
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_model.fit(X_train, y_train)
ada_predictions = ada_model.predict(X_test)
ada_accuracy = accuracy_score(y_test, ada_predictions)
print(f"Boosting (AdaBoost) Accuracy: {ada_accuracy:.2f}")
