import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Load corrected interaction data
# --------------------------------------------------

df = pd.read_csv("ltr_interactions_corrected.csv")

print("Corrected LTR Dataset Loaded Successfully!")
print()


# Features used by the ranker
FEATURES = [
    "relevance_feature",
    "distance_score",
    "rating_score"
]


# --------------------------------------------------
# Split queries into train and test
# --------------------------------------------------

query_ids = df["query_id"].unique()

train_queries, test_queries = train_test_split(
    query_ids,
    test_size=0.2,
    random_state=42
)

train_df = df[df["query_id"].isin(train_queries)].copy()
test_df = df[df["query_id"].isin(test_queries)].copy()

print("Training Queries:", len(train_queries))
print("Testing Queries:", len(test_queries))
print()


# --------------------------------------------------
# Create pairwise training examples
# --------------------------------------------------

pairwise_X = []
pairwise_y = []
pairwise_weights = []


for query_id, group in train_df.groupby("query_id"):

    clicked_items = group[group["clicked"] == 1]
    non_clicked_items = group[group["clicked"] == 0]

    for _, positive in clicked_items.iterrows():

        for _, negative in non_clicked_items.iterrows():

            positive_features = (
                positive[FEATURES]
                .values
                .astype(float)
            )

            negative_features = (
                negative[FEATURES]
                .values
                .astype(float)
            )

            # Positive item should rank higher
            difference = (
                positive_features - negative_features
            )

            pairwise_X.append(difference)
            pairwise_y.append(1)

            # Reverse the pair
            reverse_difference = (
                negative_features - positive_features
            )

            pairwise_X.append(reverse_difference)
            pairwise_y.append(0)

            # Use average IPW weight for the pair
            pair_weight = (
                positive["ipw_weight"]
                + negative["ipw_weight"]
            ) / 2

            pairwise_weights.append(pair_weight)
            pairwise_weights.append(pair_weight)


X = np.array(pairwise_X)
y = np.array(pairwise_y)
weights = np.array(pairwise_weights)


print("Pairwise Training Data Created!")
print("Training Pairs:", len(X))
print()


# --------------------------------------------------
# Train Pairwise LTR Model
# --------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X,
    y,
    sample_weight=weights
)


print("Pairwise LTR Model Trained Successfully!")
print()

print("Model Coefficients:")
for feature, coefficient in zip(
    FEATURES,
    model.coef_[0]
):
    print(
        f"{feature}: {coefficient:.6f}"
    )


# --------------------------------------------------
# Save model coefficients
# --------------------------------------------------

np.save(
    "ltr_model_coefficients.npy",
    model.coef_[0]
)

print()
print("LTR Model Coefficients Saved Successfully!")