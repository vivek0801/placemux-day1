from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

# Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Dummy Model
model = DummyClassifier(strategy="most_frequent")

# Train Model
model.fit(X_train, y_train)

print("Dummy Model Trained Successfully!")