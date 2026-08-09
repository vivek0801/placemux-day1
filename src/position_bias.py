import pandas as pd

# Load interaction data
df = pd.read_csv("ltr_interactions.csv")

print("LTR Dataset Loaded Successfully!")
print()

# Estimated probability of being examined/clicked because of position
df["propensity"] = 1 / df["position"]

# Inverse Propensity Weighting
df["ipw_weight"] = 1 / df["propensity"]

print("Position Bias Correction Applied!")
print()

print("Position Propensity:")
print(
    df.groupby("position")["propensity"]
    .first()
)

print()

print("Inverse Propensity Weights:")
print(
    df.groupby("position")["ipw_weight"]
    .first()
)

print()

print("Sample Corrected Data:")
print(
    df[
        [
            "query_id",
            "item_id",
            "position",
            "clicked",
            "propensity",
            "ipw_weight"
        ]
    ].head(10)
)

# Save corrected dataset
df.to_csv("ltr_interactions_corrected.csv", index=False)

print()
print("Corrected LTR Dataset Saved Successfully!")