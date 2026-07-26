import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import joblib

# Load Dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Loaded Successfully!")
print("Total Samples :", len(X))
print("Total Features:", X.shape[1])

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nDataset Split Successfully!")
print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# Create End-to-End Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("\nPipeline Created Successfully!")

# Train the Pipeline
pipeline.fit(X_train, y_train)

print("\nPipeline Trained Successfully!")

# Make Predictions
y_pred = pipeline.predict(X_test)

print("\nPrediction Completed Successfully!")

# Evaluate the Pipeline
accuracy = accuracy_score(y_test, y_pred)

print("\nPipeline Accuracy:", accuracy)

# Save the trained pipeline
joblib.dump(pipeline, "pipeline.pkl")

print("\nPipeline Saved Successfully!")

# Save metrics
metrics = pd.DataFrame({
    "Metric": ["Accuracy"],
    "Value": [accuracy]
})

metrics.to_csv("metrics.csv", index=False)

print("Metrics Saved Successfully!")

with open("experiment_log.txt", "w") as file:
    file.write("PlaceMux Day 8\n")
    file.write("--------------------------\n")
    file.write(f"Training Samples : {len(X_train)}\n")
    file.write(f"Testing Samples  : {len(X_test)}\n")
    file.write(f"Accuracy         : {accuracy:.4f}\n")

print("Experiment Log Saved Successfully!")