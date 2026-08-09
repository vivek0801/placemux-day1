import pandas as pd

# Load LTR interaction data
df = pd.read_csv("ltr_interactions.csv")

print("LTR Dataset Loaded Successfully!")
print()

print("Dataset Shape:")
print(df.shape)

print()

print("Columns:")
print(df.columns.tolist())

print()

print("Click Rate by Position:")
print(
    df.groupby("position")["clicked"]
    .mean()
)

print()

print("Number of Interactions by Position:")
print(
    df["position"].value_counts().sort_index()
)

print()

print("Overall Click Rate:")
print(df["clicked"].mean())