import joblib


MODEL_PATH = "calibrated_classifier.pkl"
FEATURE_NAMES_PATH = "feature_names.pkl"


# Load model
model = joblib.load(MODEL_PATH)

# Load feature names
feature_names = joblib.load(FEATURE_NAMES_PATH)


print("Model Loaded Successfully!")
print(
    "Model Type:",
    type(model).__name__
)

print()

print("Feature Names Loaded Successfully!")
print(
    "Number of Features:",
    len(feature_names)
)

print()

print("First 5 Features:")
print(feature_names[:5])