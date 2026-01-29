# train_model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import joblib

# Load your dataset
df = pd.read_csv("personality_synthetic_dataset.csv")

# Display basic info
print("Dataset Info:")
print(df.info())
print("\nClass distribution:")
print(df['personality_type'].value_counts())

# Encode the target variable
label_encoder = LabelEncoder()
df['personality_type_encoded'] = label_encoder.fit_transform(df['personality_type'])

# Define features and target
X = df.drop(['personality_type', 'personality_type_encoded'], axis=1)
y = df['personality_type_encoded']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Feature importance (coefficients)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': np.abs(model.coef_[0])  # Using first class coefficients
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Save the model and preprocessing objects
joblib.dump(model, 'personality_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(list(X.columns), 'feature_columns.pkl')

print("\nModel and preprocessing objects saved successfully!")
