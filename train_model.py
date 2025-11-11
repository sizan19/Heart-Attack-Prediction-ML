"""
Train the heart disease prediction model and save it as model.pkl
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("notebook and datasets/Medicaldataset.csv")

# Create a copy to work with
df = data.copy()

# Encode the target variable 'Result'
le = LabelEncoder()
if 'Result' in df.columns and df['Result'].dtype == 'object':
    df['Result'] = le.fit_transform(df['Result'])
    # 0 = 'negative', 1 = 'positive'
    label_map = {i: class_name for i, class_name in enumerate(le.classes_)}
else:
    label_map = {0: '0', 1: '1'}

# Split the data
X = df.drop("Result", axis=1)
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model (best performer)
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=20, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import accuracy_score, f1_score
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='binary')

print(f"\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save the model and label encoder
model_data = {
    'model': rf_model,
    'label_encoder': le,
    'feature_names': X.columns.tolist()
}

with open('model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\n✅ Model saved successfully as 'model.pkl'")
print(f"Features: {X.columns.tolist()}")
