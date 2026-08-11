import joblib
import numpy as np


# --------------------------------------------------
# Configuration
# --------------------------------------------------

import os

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "calibrated_classifier.pkl"
)

MODEL_VERSION = "day12-v1"

DECISION_THRESHOLD = 0.77


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Single Record Prediction
# --------------------------------------------------

def predict_single(features):
    """
    Predict the probability for a single record.

    Parameters:
        features: list or array containing 30 feature values

    Returns:
        Dictionary containing score, meaning,
        prediction, and model version.
    """

    # Validate number of features
    if len(features) != 30:
        raise ValueError(
            f"Expected 30 features, received {len(features)}"
        )

    # Convert to numpy array
    X = np.asarray(features, dtype=float).reshape(1, -1)

    # Get probability of class 1
    probability = model.predict_proba(X)[0][1]

    # Apply operating threshold
    prediction = int(
        probability >= DECISION_THRESHOLD
    )

    return {
        "score": round(float(probability), 6),
        "score_meaning": "Probability of class 1 (benign)",
        "prediction": prediction,
        "model_version": MODEL_VERSION
    }


# --------------------------------------------------
# Batch Prediction
# --------------------------------------------------

def predict_batch(records):
    """
    Predict probabilities for multiple records.

    Parameters:
        records: list of feature lists

    Returns:
        List of standardized prediction dictionaries.
    """

    if not isinstance(records, list):
        raise ValueError(
            "Batch input must be a list of records"
        )

    results = []

    for index, record in enumerate(records):

        try:
            result = predict_single(record)

            result["record_index"] = index

            results.append(result)

        except ValueError as error:

            results.append({
                "record_index": index,
                "error": str(error),
                "model_version": MODEL_VERSION
            })

    return results