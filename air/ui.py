import streamlit as st
import polars as pl
import xgboost as xgb
from air.data.load import load_final_data, load_preprocessed_data_small
from air.processing.model_predictions import (
    XGBoostModelPredictionProcessor,
    BertModelPredictionProcessor,
    XGBoostHelpfulnessPredictionProcessor,
)
from air.processing.preprocessing import HelpfulnessPredictionInputProcessor
from air.models.recommender_system import RecommenderSystem


def start(data, models, helpfullness_model, recommender_model):
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
        st.write("Some random users:")
        st.write(*users[:10])
        user_id = st.text_input("User ID")
        if user_id:
            print(user_id)
            recs = recommender.recommend_books_based_on_similar_users(user_id)
            liked = (
                data.filter(pl.col("User_id") == user_id)
                .select("Title", "review/text", "review/score")
                .sort(by="review/score", descending=True)
            )
            cols = st.columns(2)
            cols[0].write("Books read by user")
            cols[0].write(liked.to_pandas())
            cols[1].write("Recommendations:")
            cols[1].write(recs.to_pandas())

    with tab2:
        st.header("Rate my review")
        st.write(
            "Enter a review and get a recommended rating and an indicator on it's helpfullness"
        )
        review = st.text_input("Review")
        if review:
            score = int(model.predict_value(review))
            helpfulness = helpfullness_model.predict_value(review)
            st.write(f"Based on you review we suggest a rating of {score}/5 stars")
            st.write(
                f"Your review is {helpfulness:.2%} helpful to other users. (Based on a xgboost model)"
            )


if __name__ == "__main__":
    data = load_final_data().sample(fraction=0.1, seed=42)
    # Init xgboost from json file
    models = {
        "xgboost": XGBoostModelPredictionProcessor(),
        "bert-base": BertModelPredictionProcessor(fine_tuned=False),
    }
    recommender = RecommenderSystem(
        data,
        num_sim_users=20,  # lower means less conservative
        num_recommendations=10,
        rating_model="xgboost",
        helpfulness_model="xgboost_helpfulness",
    )

    start(
        data,
        models,
        helpfullness_model=XGBoostHelpfulnessPredictionProcessor(),
        recommender_model=recommender,
    )
