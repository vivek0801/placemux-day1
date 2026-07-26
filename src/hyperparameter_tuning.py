from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Loaded Successfully!")
print("Total Samples :", len(X))
print("Total Features:", X.shape[1])

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nDataset Split Successfully!")
print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# Create Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])

print("\nPipeline Created Successfully!")

# Hyperparameter Grid
param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__max_iter": [100, 500, 1000]
}

print("\nHyperparameter Grid Created!")

# Grid Search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

print("GridSearchCV Created Successfully!")

# Train Grid Search
grid_search.fit(X_train, y_train)

print("\nGrid Search Completed Successfully!")

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation Accuracy:")
print(grid_search.best_score_)

# Best Model
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:")
print(accuracy)