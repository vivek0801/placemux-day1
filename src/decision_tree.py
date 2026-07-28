from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd

# Load Dataset
data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Dataset Loaded Successfully!")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# model = DecisionTreeClassifier(random_state=42)

# model.fit(X_train, y_train)

# print("\nDecision Tree Model Trained Successfully!")

# Make Predictions
# y_pred = model.predict(X_test)

# print("\nPredictions Completed Successfully!")

# Accuracy
# accuracy = accuracy_score(y_test, y_pred)

# print("\nAccuracy:", accuracy)

depths = [1, 2, 3, 5, 10, None]

print("\nDecision Tree Comparison")

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Depth = {depth}, Accuracy = {accuracy:.4f}")
    

# Train final model with best depth
final_model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

final_model.fit(X_train, y_train)

# Feature Importance
importance = pd.DataFrame({
    "Feature": data.feature_names,
    "Importance": final_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features")
print(importance.head(10))

plt.figure(figsize=(10,6))

plt.barh(
    importance["Feature"][:10],
    importance["Importance"][:10]
)

plt.gca().invert_yaxis()

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title("Decision Tree Feature Importance")

plt.show()

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))