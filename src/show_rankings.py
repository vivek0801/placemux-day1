import pandas as pd
import numpy as np


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


# --------------------------------------------------
# Calculate LTR score
# --------------------------------------------------

df["ltr_score"] = (
    df["relevance_feature"] * coefficients[0]
    + df["distance_score"] * coefficients[1]
    + df["rating_score"] * coefficients[2]
)


# --------------------------------------------------
# Calculate heuristic baseline score
# --------------------------------------------------

df["heuristic_score"] = (
    0.5 * df["relevance_feature"]
    + 0.3 * df["rating_score"]
    + 0.2 * (1 - df["distance_score"])
)


# --------------------------------------------------
# Display rankings
# --------------------------------------------------

print("========================================")
print("LTR RANKING DEMONSTRATION")
print("========================================")
print()


# Show first 5 queries
for query_id, group in df.groupby("query_id"):

    if query_id > 5:
        break

    print(f"Query {query_id}")
    print("-" * 40)

    # LTR ranking
    ltr_ranking = group.sort_values(
        "ltr_score",
        ascending=False
    )

    print("LTR Ranking:")

    for rank, (_, row) in enumerate(
        ltr_ranking.iterrows(),
        start=1
    ):
        print(
            f"{rank}. Item {int(row['item_id'])} "
            f"| Score = {row['ltr_score']:.4f} "
            f"| Clicked = {int(row['clicked'])}"
        )

    print()

    # Baseline ranking
    baseline_ranking = group.sort_values(
        "heuristic_score",
        ascending=False
    )

    print("Heuristic Baseline Ranking:")

    for rank, (_, row) in enumerate(
        baseline_ranking.iterrows(),
        start=1
    ):
        print(
            f"{rank}. Item {int(row['item_id'])} "
            f"| Score = {row['heuristic_score']:.4f} "
            f"| Clicked = {int(row['clicked'])}"
        )

    print()
    print("=" * 50)
    print()