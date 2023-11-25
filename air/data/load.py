import polars as pl


def load_raw_data() -> pl.DataFrame:
    return pl.read_csv("air/data/Books_rating.csv")
