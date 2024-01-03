import polars as pl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from air.data.load import load_preprocessed_data_small


def recommend_books_based_on_similar_users(
    ratings, user_id, cos_sim_matrix, num_sim_users=5, num_recommendations=5
):
    user_cos_sim = cos_sim_matrix.filter(pl.col("User_id") == user_id).drop("User_id")
    # We aggregate the cosine similarity of the user with all other users
    # by taking the sum of the cosine similarity of the user with each user
    # Reviews of the same user (or read books) are dropped later
    user_cos_sim = user_cos_sim.sum().to_numpy().flatten()
    # We don't want to base recommendations on reviews of the same user
    same_user_indices = cos_sim_matrix.select("User_id").to_numpy().flatten() == user_id
    user_cos_sim[same_user_indices] = 0
    # We don't want to recommend books that the user already read
    read_books = (
        ratings.filter(pl.col("User_id") == user_id).select("Title").to_series(0)
    )
    read_books_indices = (
        ratings.select("Title")
        .to_series(0)
        .is_in(read_books)
        .to_numpy()
        .flatten()
        .astype(bool)
    )
    user_cos_sim[read_books_indices] = 0

    # The idea is that people that review books similar
    # might like the same books
    top_n_indices = np.argsort(user_cos_sim)[-num_sim_users - 1 : -1]
    top_n_sim_users = cos_sim_matrix[top_n_indices].select("User_id").to_series(0)
    filtered_ratings = ratings.filter(pl.col("User_id").is_in(top_n_sim_users))
    # Count Books of users that reviewd similar to the given user
    book_rating_sum = (
        filtered_ratings.group_by(by="Title")
        .agg((pl.col("review/score") - 3).sum().alias("sum"))
        .sort(by="sum")
    )
    return book_rating_sum.top_k(num_recommendations, by="sum").select("Title")


data = load_preprocessed_data_small()
print(data.columns)

data = data.drop_nulls(subset=["User_id", "Title"])

# Filter users with more than 200 reviews
# 200 reviews returns 335 users
# 300 reviews returns 166 users
filtered_users = data.group_by(by="User_id").count().filter(pl.col("count") > 200)
filtered_books = data.group_by(by="Title").count().filter(pl.col("count") > 50)


relevant_user_ratings = data.join(filtered_users, "User_id", "inner").drop("count")
ratings = relevant_user_ratings.join(filtered_books, "Title", "inner").drop("count")

# Textual part
# Create tfidf matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(
    ratings.select("preprocessed_review/text").to_numpy().flatten()
)
cos_sim_matrix = cosine_similarity(tfidf_matrix)
cos_sim_user_matrix = pl.DataFrame(cos_sim_matrix)
cos_sim_user_matrix = cos_sim_user_matrix.with_columns(
    ratings["User_id"].alias("User_id")
)

users = ratings.select("User_id").unique().to_numpy().flatten()
my_user = users[0]

res = recommend_books_based_on_similar_users(
    ratings, my_user, cos_sim_user_matrix, 5, 5
)
print(res)


# Liked by user
liked = (
    ratings.filter(pl.col("User_id") == my_user)
    .sort(by="review/score")
    .select("Title", "review/score")
)
print(liked)
