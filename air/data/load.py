import polars as pl


def load_raw_data() -> pl.DataFrame:
    return pl.read_csv("air/data/Books_rating.csv")


def load_raw_data_small() -> pl.DataFrame:
    # Only take 10% of the data
    frame = pl.read_csv("air/data/Books_rating.csv")
    return frame.sample(fraction=0.1, seed=42)


def load_preprocessed_data() -> pl.DataFrame:
    return pl.read_ipc("air/data/checkpoints/Review Preprocessing Full.ipc")


def load_preprocessed_data_small() -> pl.DataFrame:
    return pl.read_ipc("air/data/checkpoints/Review Preprocessing.ipc")
