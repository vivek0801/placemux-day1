import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

data = load_breast_cancer()

X = data.data
y = data.target

feature_names = data.feature_names

print("Dataset Loaded Successfully!")


# --------------------------------------------------
# Train / Validation / Test Split
# --------------------------------------------------

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


# --------------------------------------------------
# Train Baseline Model
# --------------------------------------------------

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

model.fit(X_train, y_train)

y_val_prob = model.predict_proba(X_val)[:, 1]


# --------------------------------------------------
# Create Validation DataFrame
# --------------------------------------------------

val_df = pd.DataFrame(
    X_val,
    columns=feature_names
)

val_df["target"] = y_val
val_df["probability"] = y_val_prob


# --------------------------------------------------
# Segment Evaluation Function
# --------------------------------------------------

def evaluate_segment(df, segment_name):

    y_true = df["target"]
    y_prob = df["probability"]

    y_pred = (y_prob >= 0.77).astype(int)

    print(f"\nSegment: {segment_name}")
    print("--------------------------------")

    print("Samples  :", len(df))
    print(
        "Accuracy :",
        f"{accuracy_score(y_true, y_pred):.4f}"
    )
    print(
        "Precision:",
        f"{precision_score(y_true, y_pred, zero_division=0):.4f}"
    )
    print(
        "Recall   :",
        f"{recall_score(y_true, y_pred, zero_division=0):.4f}"
    )
    print(
        "F1 Score :",
        f"{f1_score(y_true, y_pred, zero_division=0):.4f}"
    )

    if len(df["target"].unique()) == 2:
        print(
            "ROC-AUC  :",
            f"{roc_auc_score(y_true, y_prob):.4f}"
        )
    else:
        print("ROC-AUC  : Not available")


# --------------------------------------------------
# Mean Radius Segments
# --------------------------------------------------

radius_feature = "mean radius"

radius_median = val_df[radius_feature].median()

low_radius = val_df[
    val_df[radius_feature] <= radius_median
]

high_radius = val_df[
    val_df[radius_feature] > radius_median
]

print("\n========================================")
print("SEGMENT EVALUATION: MEAN RADIUS")
print("========================================")

print(f"Median: {radius_median:.4f}")

evaluate_segment(
    low_radius,
    "Low Mean Radius"
)

evaluate_segment(
    high_radius,
    "High Mean Radius"
)


# --------------------------------------------------
# Mean Texture Segments
# --------------------------------------------------

texture_feature = "mean texture"

texture_median = val_df[texture_feature].median()

low_texture = val_df[
    val_df[texture_feature] <= texture_median
]

high_texture = val_df[
    val_df[texture_feature] > texture_median
]

print("\n========================================")
print("SEGMENT EVALUATION: MEAN TEXTURE")
print("========================================")

print(f"Median: {texture_median:.4f}")

evaluate_segment(
    low_texture,
    "Low Mean Texture"
)

evaluate_segment(
    high_texture,
    "High Mean Texture"
)


print("\nSegment Evaluation Completed Successfully!")