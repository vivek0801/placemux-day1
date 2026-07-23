from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Dataset Loaded and Split Successfully!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Train the baseline model
dummy_model = DummyClassifier(strategy="most_frequent")

dummy_model.fit(X_train, y_train)

# Make predictions
dummy_predictions = dummy_model.predict(X_test)

# Calculate accuracy
dummy_accuracy = accuracy_score(y_test, dummy_predictions)

print("\nBaseline Model (DummyClassifier)")
print("Accuracy:", dummy_accuracy)

# Train the real Machine Learning model
logistic_model = LogisticRegression(max_iter=200)

logistic_model.fit(X_train, y_train)

# Make predictions
logistic_predictions = logistic_model.predict(X_test)

# Calculate accuracy
logistic_accuracy = accuracy_score(y_test, logistic_predictions)

print("\nLogistic Regression Model")
print("Accuracy:", logistic_accuracy)

# Compare both models
print("\nModel Comparison")

if logistic_accuracy > dummy_accuracy:
    print("✅ Logistic Regression performed better than the baseline.")
else:
    print("❌ Logistic Regression did not outperform the baseline.")
    



# Inspect incorrect predictions
print("\nIncorrect Predictions:")

for i in range(len(y_test)):
    if y_test[i] != logistic_predictions[i]:
        print(
            f"Sample {i}: Actual = {y_test[i]}, Predicted = {logistic_predictions[i]}"
        )