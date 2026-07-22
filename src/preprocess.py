import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Pipeline for numerical features

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, [0, 1, 2, 3])
    ]
)

from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

processed_data = preprocessor.fit_transform(df)

print("Preprocessing Completed Successfully!\n")
print(processed_data[:5])

joblib.dump(preprocessor, "preprocessor.pkl")

print("\nPreprocessor Saved Successfully!")