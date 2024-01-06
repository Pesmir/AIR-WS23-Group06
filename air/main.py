import polars as pl
from air.models.recommender_system import RecommenderSystem
from air.processing.preprocessing import (
    DropEmptyReviewsAndUsers,
    DropNotNeededColumns,
    HelpfulnessPredictionInputProcessor,
    PreprocessingFinalizer,
    ReviewPreprocessor,
    DropNullReviewsPreprocessor,
    HelpfulnessPreprocessor,
    FilterUsersWithLessThen50Reviews,
    SplitReviewTokens,
)

from air.processing.model_predictions import (
    XGBoostHelpfulnessPredictionProcessor,
    XGBoostModelPredictionProcessor,
    BertModelPredictionProcessor,
)

# from air.data.load import load_raw_data
from air.data.load import load_raw_data


def call_recommender_system(data: pl.DataFrame):
    selected_user = "A1CGLIDN7E5MK8"
    recommender = RecommenderSystem(
        data,
        num_sim_users=5,  # lower means less conservative
        num_recommendations=8,
        rating_model="xgboost",
        helpfulness_model="xgboost_helpfulness",
    )
    recs = recommender.recommend_books_based_on_similar_users(selected_user)
    liked = (
        data.filter(pl.col("User_id") == selected_user)
        .sort(by="review/score")
        .select("Title", "review/score")
        .filter(pl.col("review/score") == 5.0)
        .sample(n=8)
    )

    print("RECOMMENDED")
    print(recs)
    print("5 STAR BOOKS BY USER (Random 8 samples)")
    print(liked)


def main():
    data = load_raw_data()
    # 1. Preprocessing
    res = DropNotNeededColumns().process(data)
    res = DropEmptyReviewsAndUsers().process(res)
    res = FilterUsersWithLessThen50Reviews().process(res)
    res = ReviewPreprocessor().process(res)
    res = DropNullReviewsPreprocessor().process(res)
    res = HelpfulnessPreprocessor().process(res)
    res = SplitReviewTokens().process(res)
    res = HelpfulnessPredictionInputProcessor().process(res)
    res = PreprocessingFinalizer().process(res)
    # The models were trainend beforehand and the corresponding files are
    # loaded in air/processing/model_predictions.py

    # 2. Add Model predictions
    res = XGBoostModelPredictionProcessor().process(res)
    # res = BertModelPredictionProcessor(fine_tuned=False).process(res)
    # res = BertModelPredictionProcessor(fine_tuned=True).process(res)
    # res = Word2VecModelPredictionProcessor().process(res)

    # 2.5 Add Helpfulnes prediction
    res = XGBoostHelpfulnessPredictionProcessor().process(res)

    # 3. Recommend book by user
    call_recommender_system(res)


if __name__ == "__main__":
    main()
