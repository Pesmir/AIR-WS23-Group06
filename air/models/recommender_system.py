import polars as pl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, re
from sklearn.metrics.pairwise import cosine_similarity

from air.data.load import load_preprocessed_data


# Put the code above into one class
class RecommenderSystem:
    def __init__(
        self,
        data,
        num_sim_users=5,
        num_recommendations=5,
        rating_model="xgboost",
        helpfulness_model="xgboost_helpfulness",
    ):
        self.data = data
        self.rating_model = rating_model
        self.helpfulness_model = helpfulness_model
        self.num_sim_users = num_sim_users
        self.num_recommendations = num_recommendations
        self.cos_sim_user_matrix = None
        self.vectorizer = TfidfVectorizer()

    def _filter_books(self, user_id):
        ratings = (
            self.data.filter(pl.col("Title").is_not_null())
            .filter(pl.col("preprocessed_review/text").is_not_null())
            .filter(pl.col("User_id").is_not_null())
        )
        # Only books with more than 50 reviews are considered
        books_reviewed_by_user = (
            ratings.filter(pl.col("User_id") == user_id).select("Title").to_series(0)
        )
        return ratings.filter(pl.col("Title").is_in(books_reviewed_by_user))

    def _create_cos_sim_matrices(self, ratings):
        books = ratings.select("Title").unique().to_numpy().flatten()
        cos_sim_matrices = {}
        for book in books:
            df = ratings.filter(pl.col("Title") == book)
            # Drop duplicated users, no idea why there are duplicated users
            df = df.unique(subset=["User_id"])
            if len(ratings) < 2:
                continue
            tfidf_matrix = self.vectorizer.fit_transform(
                df.select("preprocessed_review/text").to_numpy().flatten()
            )
            user_cols = df.select("User_id").to_numpy().flatten()
            cos_sim_matrix = cosine_similarity(tfidf_matrix)
            cos_sim_matrix = pl.DataFrame(cos_sim_matrix)
            cos_sim_matrix.columns = user_cols
            cos_sim_matrices[book] = cos_sim_matrix
        return cos_sim_matrices

    def _compute_user_sim_scores(self, cos_sim_matrices, user_id):
        user_scores = {}
        num_books = len(cos_sim_matrices)
        for book in cos_sim_matrices:
            cos_sim_frame = cos_sim_matrices[book]
            cos_sim = cos_sim_frame.select(user_id).to_numpy().flatten()
            users = cos_sim_frame.columns
            for i in range(len(users)):
                user = users[i]
                if user not in user_scores:
                    user_scores[user] = 0
                else:
                    user_scores[user] += cos_sim[i] / num_books
        # We don't want to recommend books that the user already read
        del user_scores[user_id]

        return pl.from_dict(user_scores).transpose(
            include_header=True,
            header_name="User_id",
            column_names=["score"],
        )

    def recommend_books_based_on_similar_users(self, user_id):
        user_books = self._filter_books(user_id)
        cos_sim_matrices = self._create_cos_sim_matrices(user_books)
        # user_scores holds the sum of the sim score for each user
        # divided by the number of books read by the selected user
        user_scores_df = self._compute_user_sim_scores(cos_sim_matrices, user_id)
        # Get top n users
        top_n_users = (
            user_scores_df.top_k(self.num_sim_users, by="score")
            .select("User_id")
            .to_series(0)
        )
        rating_agg_expr = (
            pl.col(self.rating_model)
            # In order to have a score between -2 and +2 we substract 3
            .sub(3)
            # We want to weigh ratings with a higher helpfulness
            # stronger
            .mul(pl.col(self.helpfulness_model))
            .mean()
            .alias("book_recommendation_score")
        )

        return (
            self.data.filter(pl.col("User_id").is_in(top_n_users))
            .group_by(by="Title")
            .agg(rating_agg_expr)
            .sort(by="book_recommendation_score")
            .top_k(self.num_recommendations, by="book_recommendation_score")
            .select("Title")
        )


if __name__ == "__main__":
    df = load_preprocessed_data()
    filtered_users = (
        df.group_by(by="User_id")
        .count()
        .filter(pl.col("count") > 5)
        .select("User_id")
        .sort(by="User_id")
        .to_numpy()
        .flatten()
    )
    my_user = filtered_users[237]
    print(f"User id: {my_user}")

    # Liked by user
    print("5 STAR BOOKS BY USER")
    liked = (
        df.filter(pl.col("User_id") == my_user)
        .sort(by="review/score")
        .select("Title", "review/score")
        .filter(pl.col("review/score") > 4.5)
    )
    print(liked)

    recommender = RecommenderSystem(df, 30, 8, rating_model="review/score")
    res = recommender.recommend_books_based_on_similar_users(my_user)
    print("RECOMMENDED")
    print(res)
