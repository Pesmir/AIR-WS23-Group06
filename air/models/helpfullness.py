from air.data.load import load_preprocessed_data, load_preprocessed_data_small
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
import random
import os
import pickle
import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def get_targets(series: pl.Series) -> np.ndarray:
    return series.to_numpy().flatten()


# Vecorising input data
input_df = load_preprocessed_data()
input_col = "input/helpfulnessprediction"
output_col = "review/helpfulness"

input_df = input_df.select([input_col, output_col]).with_row_count("row_nr")
# We throw away 0 and 1 helpfulness scores
input_df = input_df.filter(pl.col(output_col) < 0.9999)
input_df = input_df.filter(pl.col(output_col) > 0.0001)
# Create input col
# Split data
test_data = input_df.sample(fraction=0.2, seed=123, shuffle=True)
val_data = test_data.sample(fraction=0.5, seed=123, shuffle=True)
train_data = input_df.filter(pl.col("row_nr").is_in(test_data["row_nr"]).not_())
test_data = test_data.filter(pl.col("row_nr").is_in(val_data["row_nr"]).not_())

print("Transforming data")
train_x = np.stack(train_data[input_col].to_numpy())
test_x = np.stack(test_data[input_col].to_numpy())
val_x = np.stack(val_data[input_col].to_numpy())
train_y = get_targets(train_data[output_col])
test_y = get_targets(test_data[output_col])
val_y = get_targets(val_data[output_col])


# Train the model

xgb_train = xgb.DMatrix(train_x, label=train_y)
xgb_test = xgb.DMatrix(test_x, label=test_y)
xgb_val = xgb.DMatrix(val_x, label=val_y)

# We use the same architecture as in the xgboost notebook
params = {
    "eta": 0.9,  # default 0.3
    "max_depth": 30,
    "objective": "reg:pseudohubererror",
    "eval_metric": ["mae", "rmse", "mphe"],
}

print("Training model")
model_path = "air/data/models/xgboost_helpfulness.pkl"
if not os.path.exists(model_path):
    xgb_model = xgb.train(
        params,
        xgb_train,
        100,
        verbose_eval=1,
        evals=[(xgb_val, "val")],
        num_boost_round=10,
    )
    xgb_model.save_model(model_path)
    with open(model_path, "wb") as f:
        pickle.dump(xgb_model, f)
else:
    with open(model_path, "rb") as f:
        xgb_model = pickle.load(f)


# Test the predictions
y_pred = xgb_model.predict(xgb_test)
test_y = test_y
# Rate of correct predictions
baseline1_pred = np.ones(len(test_y))
baseline0_pred = np.zeros(len(test_y))
baseline0_5_pred = baseline1_pred * 0.5

# Calculate the MAE for baseline and predictions
print("Baseline 1 MAE: ", np.sum(np.abs(baseline1_pred - test_y)) / len(test_y))
print("Baseline 0.5 MAE: ", np.sum(np.abs(baseline0_5_pred - test_y)) / len(test_y))
print("Baseline 0 MAE: ", np.sum(np.abs(baseline0_pred - test_y)) / len(test_y))
print("Prediction MAE: ", np.sum(np.abs(y_pred - test_y)) / len(test_y))

print(
    "Baseline 1 MSE: ",
    np.sum(np.sqrt(np.square(baseline1_pred - test_y))) / len(test_y),
)
print(
    "Baseline 0.5 MSE: ",
    np.sum(np.sqrt(np.square(baseline0_5_pred - test_y))) / len(test_y),
)
print(
    "Baseline 0 MSE: ",
    np.sum(np.sqrt(np.square(baseline0_pred - test_y))) / len(test_y),
)
print("Prediction MSE: ", np.sum(np.sqrt(np.square(y_pred - test_y))) / len(test_y))

sns.histplot(y_pred)
sns.histplot(test_y)
plt.show()
plt.scatter(y_pred, test_y)
plt.show()

print(y_pred)
print(test_y)
