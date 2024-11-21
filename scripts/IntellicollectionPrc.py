# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample data: historical payment behavior
data = {
    'User_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Trips': [5, 3, 6, 2, 8, 7, 4, 5, 6, 3],
    'Total_Amount': [100, 60, 120, 40, 160, 140, 80, 100, 120, 60],
    'Paid': [1, 0, 1, 0, 1, 1, 0, 1, 1, 0]  # 1 for paid, 0 for not paid
}

# Convert data to a DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df[['Trips', 'Total_Amount']]
y = df['Paid']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
