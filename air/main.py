from air.processing.preprocessing import (
    ReviewPreprocessor,
    DropNullReviewsPreprocessor,
    HelpfulnessPreprocessor,
)

# from air.data.load import load_raw_data
from air.data.load import load_raw_data


def main():
    data = load_raw_data()
    res = ReviewPreprocessor().process(data)
    res = DropNullReviewsPreprocessor().process(res)
    res = HelpfulnessPreprocessor().process(res)
    print(res.head())
    print(res.columns)


if __name__ == "__main__":
    main()
