import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the dataset
data = pd.read_csv('ml_model/dataset.csv')  # <-- updated relative path

# Split features and labels
X = data[['length', 'semicolons', 'parentheses']]
y = data['label']

# Train the model
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Save the model
joblib.dump(clf, 'ml_model/model.pkl')

print("✅ Model trained and saved successfully.")
