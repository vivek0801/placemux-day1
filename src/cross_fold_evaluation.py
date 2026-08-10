import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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
print(f"Total Samples : {X.shape[0]}")
print(f"Total Features: {X.shape[1]}")


# --------------------------------------------------
# Cross Validation Setup
# --------------------------------------------------

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# --------------------------------------------------
# Store Results
# --------------------------------------------------

results = []


# --------------------------------------------------
# Cross-Fold Evaluation
# --------------------------------------------------

for fold, (train_idx, val_idx) in enumerate(
    skf.split(X, y), start=1
):

    X_train = X[train_idx]
    X_val = X[val_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]


    # Pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])


    # Train
    model.fit(X_train, y_train)


    # Predictions
    y_pred = model.predict(X_val)

    y_prob = model.predict_proba(X_val)[:, 1]


    # Metrics
    accuracy = accuracy_score(y_val, y_pred)

    precision = precision_score(
        y_val,
        y_pred
    )

    recall = recall_score(
        y_val,
        y_pred
    )

    f1 = f1_score(
        y_val,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_val,
        y_prob
    )

    loss = log_loss(
        y_val,
        y_prob
    )


    results.append({
        "Fold": fold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "Log Loss": loss
    })


# --------------------------------------------------
# Results
# --------------------------------------------------

results_df = pd.DataFrame(results)


print("\n========================================")
print("5-FOLD CROSS VALIDATION RESULTS")
print("========================================")

print(results_df.to_string(index=False))


# --------------------------------------------------
# Mean and Standard Deviation
# --------------------------------------------------

print("\n========================================")
print("STABILITY SUMMARY")
print("========================================")

for metric in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
    "Log Loss"
]:

    mean_value = results_df[metric].mean()
    std_value = results_df[metric].std()

    print(
        f"{metric:<10}: "
        f"Mean = {mean_value:.4f}, "
        f"Std = {std_value:.4f}"
    )


print("\nCross-Fold Evaluation Completed Successfully!")