from air.processing.preprocessing import (
    ReviewPreprocessor,
    DropNullReviewsPreprocessor,
    HelpfulnessPreprocessor,
)

from air.processing.model_predictions import (
    XGBoostModelPredictionProcessor,
    BertModelPredictionProcessor,
)

# from air.data.load import load_raw_data
from air.data.load import load_raw_data


def main():
    data = load_raw_data()
    # 1. Preprocessing
    res = ReviewPreprocessor().process(data)
    res = DropNullReviewsPreprocessor().process(res)
    res = HelpfulnessPreprocessor().process(res)
    # The models where trainend beforehand and the corresponding files are
    # loaded in air/processing/model_predictions.py

    # 2. Model predictions
    res = XGBoostModelPredictionProcessor().process(res)
    res = BertModelPredictionProcessor(fine_tuned=False).process(res)

    print(res.head())
    print(res.columns)


if __name__ == "__main__":
    main()
