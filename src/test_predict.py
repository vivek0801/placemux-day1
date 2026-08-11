from schemas import PredictionInput, BatchPredictionInput
from predict import predict_single, predict_batch
from sklearn.datasets import load_breast_cancer

# Load real dataset
data = load_breast_cancer()

# --------------------------------------------------
# Test Single Prediction
# --------------------------------------------------

single_features = data.data[0].tolist()

try:

    validated_input = PredictionInput(
        features=single_features
    )

    result = predict_single(
        validated_input.features
    )

    print("========================================")
    print("SINGLE PREDICTION (Real Data)")
    print("========================================")

    print(result)

except Exception as error:

    print("Single Prediction Failed:")
    print(error)


# --------------------------------------------------
# Test Batch Prediction
# --------------------------------------------------

batch_features = [
    data.data[0].tolist(),
    data.data[1].tolist(),
    data.data[2].tolist()
]

try:

    validated_batch = BatchPredictionInput(
        records=[
            PredictionInput(features=record)
            for record in batch_features
        ]
    )

    results = predict_batch(
        [
            record.features
            for record in validated_batch.records
        ]
    )

    print("\n========================================")
    print("BATCH PREDICTION")
    print("========================================")

    for result in results:
        print(result)

except Exception as error:

    print("Batch Prediction Failed:")
    print(error)
    

# --------------------------------------------------
# Test Invalid Single Input
# --------------------------------------------------

print("\n========================================")
print("INVALID INPUT TEST")
print("========================================")

invalid_features = [1.0] * 5

try:

    invalid_input = PredictionInput(
        features=invalid_features
    )

    result = predict_single(
        invalid_input.features
    )

    print("ERROR: Invalid input was accepted!")
    print(result)

except Exception as error:

    print("Invalid Input Rejected Successfully!")
    print("Error:", error)