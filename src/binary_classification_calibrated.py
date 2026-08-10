import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import (
    CalibratedClassifierCV,
    CalibrationDisplay
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Loaded Successfully!")
print("Total Samples:", len(X))
print("Total Features:", X.shape[1])


# --------------------------------------------------
# Train / Validation / Test Split
# --------------------------------------------------

# First: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Second: split remaining 80% into train and validation
# 25% of 80% = 20% of the complete dataset
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

print("\nDataset Split Successfully!")

print("Training Samples  :", len(X_train))
print("Validation Samples:", len(X_val))
print("Testing Samples   :", len(X_test))

# --------------------------------------------------
# Baseline Logistic Regression
# --------------------------------------------------

baseline_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# Train on training data only
baseline_model.fit(X_train, y_train)

print("\nBaseline Model Trained Successfully!")


# --------------------------------------------------
# Validation Predictions
# --------------------------------------------------

y_val_pred = baseline_model.predict(X_val)
y_val_prob = baseline_model.predict_proba(X_val)[:, 1]


# --------------------------------------------------
# Baseline Validation Metrics
# --------------------------------------------------

accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)
roc_auc = roc_auc_score(y_val, y_val_prob)
loss = log_loss(y_val, y_val_prob)

print("\nBaseline Validation Results")
print("--------------------------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"Log Loss : {loss:.4f}")

# --------------------------------------------------
# Calibrated Logistic Regression
# --------------------------------------------------

calibrated_model = CalibratedClassifierCV(
    estimator=baseline_model,
    method="sigmoid",
    cv=5
)

calibrated_model.fit(X_train, y_train)

print("\nCalibrated Model Trained Successfully!")


# --------------------------------------------------
# Calibrated Validation Predictions
# --------------------------------------------------

y_cal_prob = calibrated_model.predict_proba(X_val)[:, 1]

# Use the default 0.5 threshold for now
y_cal_pred = (y_cal_prob >= 0.5).astype(int)


# --------------------------------------------------
# Calibrated Validation Metrics
# --------------------------------------------------

cal_accuracy = accuracy_score(y_val, y_cal_pred)
cal_precision = precision_score(y_val, y_cal_pred)
cal_recall = recall_score(y_val, y_cal_pred)
cal_f1 = f1_score(y_val, y_cal_pred)
cal_roc_auc = roc_auc_score(y_val, y_cal_prob)
cal_log_loss = log_loss(y_val, y_cal_prob)

print("\nCalibrated Validation Results")
print("--------------------------------")
print(f"Accuracy : {cal_accuracy:.4f}")
print(f"Precision: {cal_precision:.4f}")
print(f"Recall   : {cal_recall:.4f}")
print(f"F1 Score : {cal_f1:.4f}")
print(f"ROC-AUC  : {cal_roc_auc:.4f}")
print(f"Log Loss : {cal_log_loss:.4f}")

# --------------------------------------------------
# Calibration Curve
# --------------------------------------------------

print("\nGenerating Calibration Curve...")

fig, ax = plt.subplots(figsize=(8, 6))

CalibrationDisplay.from_predictions(
    y_val,
    y_cal_prob,
    n_bins=10,
    strategy="uniform",
    ax=ax,
    name="Calibrated Logistic Regression"
)

ax.set_title("Calibration Curve")
ax.grid(True)

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Cost-Optimal Threshold Selection
# --------------------------------------------------

FP_COST = 10
FN_COST = 1

thresholds = np.arange(0.05, 0.96, 0.01)

best_threshold = None
lowest_cost = float("inf")

threshold_results = []

for threshold in thresholds:

    y_threshold_pred = (y_cal_prob >= threshold).astype(int)

    # Confusion matrix manually calculated
    fp = np.sum((y_val == 0) & (y_threshold_pred == 1))
    fn = np.sum((y_val == 1) & (y_threshold_pred == 0))

    total_cost = (
        FP_COST * fp
        + FN_COST * fn
    )

    threshold_results.append(
        (threshold, fp, fn, total_cost)
    )

    if total_cost < lowest_cost:
        lowest_cost = total_cost
        best_threshold = threshold


print("\nCost-Optimal Threshold")
print("--------------------------------")
print(f"False Positive Cost: {FP_COST}")
print(f"False Negative Cost: {FN_COST}")
print(f"Best Threshold     : {best_threshold:.2f}")
print(f"Minimum Cost       : {lowest_cost}")


# --------------------------------------------------
# Evaluate at Optimal Threshold
# --------------------------------------------------

y_optimal_pred = (
    y_cal_prob >= best_threshold
).astype(int)

optimal_accuracy = accuracy_score(
    y_val,
    y_optimal_pred
)

optimal_precision = precision_score(
    y_val,
    y_optimal_pred
)

optimal_recall = recall_score(
    y_val,
    y_optimal_pred
)

optimal_f1 = f1_score(
    y_val,
    y_optimal_pred
)

print("\nPerformance at Optimal Threshold")
print("--------------------------------")
print(f"Accuracy : {optimal_accuracy:.4f}")
print(f"Precision: {optimal_precision:.4f}")
print(f"Recall   : {optimal_recall:.4f}")
print(f"F1 Score : {optimal_f1:.4f}")