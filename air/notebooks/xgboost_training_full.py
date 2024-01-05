from air.data.load import load_preprocessed_data
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
import os
import pickle
import polars as pl
import numpy as np


def get_targets(series) -> np.ndarray:
    return series.to_numpy().astype(int) - 1


# Vecorising input data
input_df = load_preprocessed_data()
input_col = "preprocessed_review/text"
output_col = "review/score"

input_df = input_df.select([input_col, output_col]).with_row_count("row_nr")
# Convert review scores to integers
input_df = input_df.with_columns(pl.col(output_col).cast(pl.Int8))

# plot histogram of review scores
# plt.hist(input_df[output_col].to_numpy().astype(str))
# plt.show()
# The histogram shows that the majority of reviews are positive (4 and 5 stars)
# We balance the dataset so that we have an equal amount of each class
# 1. Find out the smallest class
smallest_class_count = (
    input_df.group_by(output_col).count().select("count").min().item()
)
# 2. Sample each class "smallest_class_count" times and concat them together
print(input_df.groupby(output_col).agg(pl.count("*")))
new_input_df = pl.DataFrame()
for target_class in input_df[output_col].unique().to_numpy():
    print(target_class)
    class_df = input_df.filter(pl.col(output_col) == target_class)
    class_df = class_df.sample(n=smallest_class_count, seed=123, shuffle=True)
    new_input_df = new_input_df.vstack(class_df)
input_df = new_input_df

# 3. Make a 80/20 split
test_data = input_df.sample(fraction=0.4, seed=123, shuffle=True)
val_data = test_data.sample(fraction=0.5, seed=123, shuffle=True)
train_data = input_df.filter(pl.col("row_nr").is_in(test_data["row_nr"]).not_())
test_data = test_data.filter(pl.col("row_nr").is_in(val_data["row_nr"]).not_())

if not os.path.exists("air/data/models/xgboost_countvectorizer_full.pkl"):
    count_vectorizer = CountVectorizer(binary=True)
    count_vectorizer.fit(input_df[input_col])

    # Save the vectorizer
    with open("air/data/models/xgboost_countvectorizer_full.pkl", "wb") as f:
        pickle.dump(count_vectorizer, f)
else:
    with open("air/data/models/xgboost_countvectorizer_full.pkl", "rb") as f:
        count_vectorizer = pickle.load(f)


print("Transforming data")
train_x = count_vectorizer.transform(train_data[input_col])
test_x = count_vectorizer.transform(test_data[input_col])
val_x = count_vectorizer.transform(val_data[input_col])
train_y = get_targets(train_data[output_col])
test_y = get_targets(test_data[output_col])
val_y = get_targets(val_data[output_col])


# Train the model

xgb_train = xgb.DMatrix(train_x, label=train_y, enable_categorical=True)
xgb_test = xgb.DMatrix(test_x, label=test_y, enable_categorical=True)
xgb_val = xgb.DMatrix(val_x, label=val_y, enable_categorical=True)

param = {
    "eta": 0.9,
    "max_depth": 5,
    "subsample": 0.1,
    "grow_policy": "lossguide",
    "num_parallel_tree": 3,
    "objective": "multi:softmax",
    "num_class": 5,
    "max_cat_to_onehot": 5,
    "eval_metric": ["mlogloss", "merror"],
}

print("Training model")
xgb_model = xgb.train(
    param, xgb_train, 50, verbose_eval=1, evals=[(xgb_val, "val")], num_boost_round=20
)

xgb_model.save_model("air/data/models/xgboost_full.json")

# Test the predictions
y_pred = xgb_model.predict(xgb_test).astype(int)
test_y = test_y.astype(int)
# Rate of correct predictions
print("Rate of correct predictions: ", np.sum(y_pred == test_y) / len(test_y))
print("Absolute correct predictions: ", np.sum(y_pred == test_y), "/", len(test_y))
baseline3_pred = np.ones(len(test_y)).astype(int) * 3
baseline_rand_pred = np.random.randint(0, 5, len(test_y))

# Calculate the MAE for baseline and predictions
print("Baseline 3 MAE: ", np.sum(np.abs(baseline3_pred - test_y)) / len(test_y))
print("Baseline rand MAE: ", np.sum(np.abs(baseline_rand_pred - test_y)) / len(test_y))
print("Prediction MAE: ", np.sum(np.abs(y_pred - test_y)) / len(test_y))

print(y_pred)
print(test_y)
