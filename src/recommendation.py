import pandas as pd
import joblib

from sklearn.metrics.pairwise import cosine_similarity


def build_recommendation_model(df):

    customer_product_matrix = pd.pivot_table(
        df,
        index="CustomerID",
        columns="Description",
        values="Quantity",
        fill_value=0
    )

    similarity = cosine_similarity(
        customer_product_matrix.T
    )

    similarity_df = pd.DataFrame(
        similarity,
        index=customer_product_matrix.columns,
        columns=customer_product_matrix.columns
    )

    joblib.dump(
        similarity_df,
        "models/item_similarity.pkl"
    )

    return similarity_df


def recommend_products(
        product_name,
        similarity_df,
        top_n=5
):

    if product_name not in similarity_df.columns:
        return []

    recommendations = (
        similarity_df[product_name]
        .sort_values(ascending=False)
        .iloc[1:top_n+1]
    )

    return recommendations.index.tolist()