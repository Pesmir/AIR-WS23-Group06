import streamlit as st
import polars as pl
import xgboost as xgb
from air.data.load import load_preprocessed_data_small
from air.processing.model_predictions import (
    XGBoostModelPredictionProcessor,
    BertModelPredictionProcessor,
)


def start(data, models):
    st.title("Book Recommender System")
    st.subheader("AIR Prject WS 2023/2024")
    st.write("This app recommend books based on previous reviews of a user")
    st.write(
        "The recommendations are based on the cosine similarity between the TF-IDF vectors of the reviews"
    )
    st.write(
        "The final rankings are computed with the help of a Model that assigns a rating to each review."
    )
    # Filter users that rated at least 200 books
    counts = (
        data.select("User_id").to_series().value_counts().filter(pl.col("counts") > 30)
    )
    # Inner join to filter out users that rated less than 200 books
    users = (
        data.join(counts, on="User_id", how="inner")
        .select("User_id")
        .unique()
        .to_numpy()
        .flatten()
    )

    # Create two tabs
    tabs = ["Recommend by user", "Rate my review"]
    selected_model = st.selectbox("Select a model", list(models.keys()), 0)
    tab1, tab2 = st.tabs(tabs)
    model = models[selected_model]

    with tab1:
        st.header("Recommend by user")
        st.write(
            "Select a user id and get a list of recommended books\n"
            "For simplicity, we only show users that rated at least 200 books"
        )
        user_id = st.selectbox("User ID", users)

        # Get recommendations
        books_read = (
            data.filter(pl.col("User_id") == user_id)
            .sort(by="review/score", descending=True)
            .select(pl.col("Title"), pl.col("review/score"))
            .to_pandas()
        )
        st.write("Books read by the user sorted by rating (best first)")
        st.write(books_read)
        st.write("Recommendations:")
        # TODO: Replace with actual recommendations
        st.write(books_read[:10])

    with tab2:
        st.header("Rate my review")
        st.write("Enter a review and get a rating")
        review = st.text_input("Review")
        if review:
            score = int(model.predict_value(review))
            st.write(f"Based on you review we suggest a rating of {score}/5 stars")


if __name__ == "__main__":
    data = load_preprocessed_data_small()

    # Init xgboost from json file
    models = {
        "xgboost": XGBoostModelPredictionProcessor(),
        "bert-base": BertModelPredictionProcessor(fine_tuned=False),
    }
    start(data, models)
