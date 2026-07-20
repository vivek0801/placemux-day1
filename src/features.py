from sklearn.datasets import load_iris
import pandas as pd

# Load Dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add Target Column
df["target"] = iris.target

print("=" * 50)
print("FEATURE PROFILING REPORT")
print("=" * 50)

# Dataset Shape
print("\n1. Dataset Shape")
print(df.shape)

# Dataset Information
print("\n2. Dataset Information")
print(df.info())

# First Five Rows
print("\n3. First Five Rows")
print(df.head())

# Statistical Summary
print("\n4. Statistical Summary")
print(df.describe())

# Missing Values
print("\n5. Missing Values")
print(df.isnull().sum())

# Target Distribution
print("\n6. Target Distribution")
print(df["target"].value_counts())