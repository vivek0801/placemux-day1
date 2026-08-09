import pandas as pd
import numpy as np

from sklearn.metrics import ndcg_score


# --------------------------------------------------
# Load data and trained model
# --------------------------------------------------

df = pd.read_csv("ltr_interactions_corrected.csv")

coefficients = np.load("ltr_model_coefficients.npy")

FEATURES = [
    "relevance_feature",
    "distance_score",
    "rating_score"
]


print("LTR Evaluation Started!")
print()


# --------------------------------------------------
# Calculate LTR model score
# --------------------------------------------------

df["ltr_score"] = (
    df["relevance_feature"] * coefficients[0]
    + df["distance_score"] * coefficients[1]
    + df["rating_score"] * coefficients[2]
)


# --------------------------------------------------
# Heuristic baseline
# --------------------------------------------------

df["heuristic_score"] = (
    0.5 * df["relevance_feature"]
    + 0.3 * df["rating_score"]
    + 0.2 * (1 - df["distance_score"])
)


# --------------------------------------------------
# Evaluation functions
# --------------------------------------------------

def calculate_ndcg(group, score_column, k):

    y_true = group["clicked"].values.reshape(1, -1)
    y_score = group[score_column].values.reshape(1, -1)

    return ndcg_score(
        y_true,
        y_score,
        k=k
    )


def calculate_map(group, score_column, k):

    ranked = group.sort_values(
        score_column,
        ascending=False
    ).head(k)

    relevant = ranked["clicked"].values

    total_relevant = group["clicked"].sum()

    if total_relevant == 0:
        return 0.0

    precision_sum = 0.0
    relevant_count = 0

    for i, value in enumerate(relevant, start=1):

        if value == 1:
            relevant_count += 1
            precision_sum += relevant_count / i

    return precision_sum / min(total_relevant, k)


# --------------------------------------------------
# Evaluate both methods
# --------------------------------------------------

ltr_ndcg_3 = []
ltr_ndcg_5 = []
ltr_map_5 = []

baseline_ndcg_3 = []
baseline_ndcg_5 = []
baseline_map_5 = []


for query_id, group in df.groupby("query_id"):

    # LTR model
    ltr_ndcg_3.append(
        calculate_ndcg(
            group,
            "ltr_score",
            3
        )
    )

    ltr_ndcg_5.append(
        calculate_ndcg(
            group,
            "ltr_score",
            5
        )
    )

    ltr_map_5.append(
        calculate_map(
            group,
            "ltr_score",
            5
        )
    )

    # Heuristic baseline
    baseline_ndcg_3.append(
        calculate_ndcg(
            group,
            "heuristic_score",
            3
        )
    )

    baseline_ndcg_5.append(
        calculate_ndcg(
            group,
            "heuristic_score",
            5
        )
    )

    baseline_map_5.append(
        calculate_map(
            group,
            "heuristic_score",
            5
        )
    )


# --------------------------------------------------
# Final results
# --------------------------------------------------

print("========================================")
print("LTR EVALUATION RESULTS")
print("========================================")
print()

print("Pairwise LTR Model:")
print(
    f"nDCG@3 : {np.mean(ltr_ndcg_3):.4f}"
)
print(
    f"nDCG@5 : {np.mean(ltr_ndcg_5):.4f}"
)
print(
    f"MAP@5  : {np.mean(ltr_map_5):.4f}"
)

print()

print("Heuristic Baseline:")
print(
    f"nDCG@3 : {np.mean(baseline_ndcg_3):.4f}"
)
print(
    f"nDCG@5 : {np.mean(baseline_ndcg_5):.4f}"
)
print(
    f"MAP@5  : {np.mean(baseline_map_5):.4f}"
)

print()
print("LTR Evaluation Completed Successfully!")