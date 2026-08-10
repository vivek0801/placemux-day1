import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


# ========================================
# Load Dataset
# ========================================

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Loaded Successfully!")


# ========================================
# Train / Validation / Test Split
# ========================================

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)


# ========================================
# Base Model
# ========================================

base_model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])


# ========================================
# Calibrated Model
# ========================================

calibrated_model = CalibratedClassifierCV(
    base_model,
    method="sigmoid",
    cv=5
)

calibrated_model.fit(X_train, y_train)

print("Calibrated Model Trained Successfully!")


# ========================================
# Save Model
# ========================================

model_path = "calibrated_classifier.pkl"

joblib.dump(
    calibrated_model,
    model_path
)

print("Model Packaged Successfully!")
print(f"Saved Model: {model_path}")


# ========================================
# Save Feature Names
# ========================================

feature_names_path = "feature_names.pkl"

joblib.dump(
    data.feature_names.tolist(),
    feature_names_path
)

print(f"Feature Names Saved: {feature_names_path}")


# ========================================
# Operating Point
# ========================================

threshold = 0.77

print("\n========================================")
print("MODEL OPERATING POINT")
print("========================================")

print(f"Decision Threshold: {threshold}")
print("False Positive Cost: 10")
print("False Negative Cost: 1")

print("\nExpected Behaviour:")
print("- Higher threshold reduces false positives.")
print("- Higher threshold can increase false negatives.")
print("- Threshold 0.77 was selected using cost minimization.")

print("\nModel Packaging Completed Successfully!")