from air.data.load import load_final_data
import polars as pl
from tqdm import tqdm

from air.models.recommender_system import RecommenderSystem

full_dataframe = load_final_data().sample(fraction=0.4, seed=42)
full_dataframe = full_dataframe.with_row_count("row_nr")
# Filter users that rated at least 100 books
# Filter only books with score 5.0
filtered_users = (
    full_dataframe.select("User_id")
    .group_by("User_id")
    .count()
    .filter(pl.col("count") > 200)
)
print(filtered_users)
dataframe = full_dataframe.join(filtered_users, on="User_id", how="inner").drop("count")
print(dataframe)
# We want to evaluate if our recommender system can
# recommend books that the user will like
test_set = dataframe.group_by("User_id").map_groups(
    lambda df: df.filter(pl.col("review/score") >= 4.0).sample(fraction=0.5, seed=42)
)

test_set_copy = (
    test_set.clone()
    .with_columns(pl.lit("ArtificialUser").alias("User_id"))
    .unique(subset="Title")
)
prediction_set = full_dataframe.filter(
    pl.col("row_nr").is_in(test_set["row_nr"]).not_()
)
prediction_set = pl.concat([prediction_set, test_set_copy])

print(len(test_set))
print(len(prediction_set))

# Make a dict where the key is the user id and the value is a set of book titles
user_best_recommendations = {}
for group in test_set.groupby("User_id"):
    user_best_recommendations[group[0]] = set(group[1]["Title"].to_numpy())


models = ["xgboost", "review/score"]  # ["xgboost", "bert-base"]

total_predictions = len(test_set["User_id"].unique())
for model in models[::-1]:
    correct_predictions = 0
    num_predictions_made = 0
    for user in tqdm(test_set["User_id"].unique().to_numpy()):
        if user is None:
            continue
        recommender = RecommenderSystem(
            prediction_set,
            num_sim_users=5,  # lower means less conservative
            num_recommendations=10,
            rating_model=model,
            helpfulness_model="xgboost_helpfulness",
        )
        recs = (
            recommender.recommend_books_based_on_similar_users(user)
            .to_numpy()
            .flatten()
        )
        recs_set = set(recs)
        num_predictions = len(recs)
        recs_set.intersection_update(user_best_recommendations[user])
        if num_predictions > 0:
            num_predictions_made += num_predictions
            correct_predictions += 1
    print(f"Model: {model}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Total predictions: {total_predictions}")
    print(f"Accuracy: {correct_predictions/total_predictions}")
