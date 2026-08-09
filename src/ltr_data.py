import pandas as pd
import numpy as np

# Reproducibility
np.random.seed(42)

# Number of queries and items per query
NUM_QUERIES = 100
ITEMS_PER_QUERY = 5

rows = []

for query_id in range(1, NUM_QUERIES + 1):

    for item_id in range(1, ITEMS_PER_QUERY + 1):

        # Simulated item/query features
        relevance_feature = np.random.uniform(0, 1)
        distance_score = np.random.uniform(0, 1)
        rating_score = np.random.uniform(0, 1)

        # Position shown to the user
        position = item_id

        # Generate underlying relevance
        relevance = (
            0.5 * relevance_feature
            + 0.3 * rating_score
            + 0.2 * distance_score
        )

        # Simulate click probability
        position_bias = 1 / position

        click_probability = relevance * position_bias

        clicked = np.random.binomial(
            1,
            min(click_probability, 0.95)
        )

        rows.append({
            "query_id": query_id,
            "item_id": item_id,
            "position": position,
            "relevance_feature": relevance_feature,
            "distance_score": distance_score,
            "rating_score": rating_score,
            "clicked": clicked
        })


# Create DataFrame
df = pd.DataFrame(rows)

# Save dataset
df.to_csv("ltr_interactions.csv", index=False)

print("LTR Interaction Dataset Created Successfully!")
print()
print("Total interactions:", len(df))
print("Total queries:", df["query_id"].nunique())
print()
print(df.head(10))