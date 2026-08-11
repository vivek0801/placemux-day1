from pydantic import ValidationError
from sklearn.datasets import load_breast_cancer

from schemas import PredictionInput


# --------------------------------------------------
# Valid Input
# --------------------------------------------------

data = load_breast_cancer()
valid_features = data.data[0].tolist()

try:

    valid_input = PredictionInput(
        features=valid_features
    )

    print("Valid Input Accepted Successfully!")
    print(
        "Number of Features:",
        len(valid_input.features)
    )

except ValidationError as error:

    print("Unexpected Validation Error:")
    print(error)


# --------------------------------------------------
# Invalid Input: Wrong Feature Count
# --------------------------------------------------

invalid_features = [1.0] * 5

try:

    PredictionInput(
        features=invalid_features
    )

    print("ERROR: Invalid input was accepted!")

except ValidationError:

    print(
        "Invalid Feature Count Rejected Successfully!"
    )
    