import polars as pl
import gc
import time
import nltk
import string
from functools import lru_cache
import multiprocessing

from air.processing.processor import Processor


stem_cnt = 0
totla_cnt = 0


class CachedPorterStemmer(nltk.PorterStemmer):
    @lru_cache(maxsize=200_000)
    def stem(self, word):
        return super().stem(word)


class ReviewPreprocessor(Processor):
    def __init__(self):
        self._num_done_iterations = 0
        self._num_total_iterations = 0
        self.to_remove = set(
            nltk.corpus.stopwords.words("english") + list(string.punctuation)
        )
        self.stemmer = CachedPorterStemmer()
        super().__init__("Review Preprocessing Full")

    def to_lower(self, data, col):
        return data.with_columns(pl.col(col).str.to_lowercase())

    def create_tokens(self, data, col):
        return data.with_columns(pl.col(col).str.split(" "))

    def remove_punctuation(self, data, col):
        return data.with_columns(pl.col(col).str.replace_all("[^\w\s]", ""))

    def remove_numbers(self, data, col):
        return data.with_columns(pl.col(col).str.replace_all("\d+", ""))

    def _stem_words(self, word_tokens):
        for idx, word in enumerate(word_tokens):
            word_tokens[idx] = self.stemmer.stem(word)
        self._num_done_iterations += 1
        if self._num_done_iterations % 1000 == 0:
            thread_name = multiprocessing.current_process().name
            print(
                f"     Thread {thread_name}: {self._num_done_iterations}/{self._num_total_iterations}"
            )
        return word_tokens

    def stem_words(self, data, col):
        return data.with_columns(pl.col(col).map_elements(self._stem_words))

    def join_tokens(self, data, col):
        return data.with_columns(pl.col(col).list.join(" "))

    def _processor_target(self, data, out_col) -> pl.DataFrame:
        print("     Starting subprocess...")
        self._num_total_iterations = len(data)

        res = (
            data.lazy()
            .pipe(self.to_lower, out_col)
            .pipe(self.remove_punctuation, out_col)
            .pipe(self.remove_numbers, out_col)
            .pipe(self.create_tokens, out_col)
            .pipe(self.stem_words, out_col)
            .pipe(self.join_tokens, out_col)
        )
        return res.collect()

    def process_value(self, value):
        # Lets built an artificaial DataFrame
        data = pl.DataFrame({"review/text": [value]})
        data = self._process_inner(data)
        return data

    def _process_inner(self, data: pl.DataFrame) -> pl.DataFrame:
        out_col = "preprocessed_review/text"
        data = data.with_columns(pl.col("review/text").alias(out_col))

        # Split data into 5 chunks
        start_time = time.time()
        data_chunks = []
        num_processes = min(10, len(data))
        if num_processes < 2:
            return self._processor_target(data, out_col)
        chunk_size = len(data) // num_processes
        for i in range(num_processes):
            data_chunks.append(data.slice(i * chunk_size, chunk_size))

        del data
        gc.collect()
        # Run in parallel
        with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
            args = [(chunk, out_col) for chunk in data_chunks]
            res = pool.starmap(self._processor_target, args)

        data = pl.concat(res)
        print(f"     Finished in {time.time() - start_time:.2f} seconds")
        return data


class DropNullReviewsPreprocessor(Processor):
    def __init__(self):
        super().__init__("Drop Null Reviews")

    def _process_inner(self, data: pl.DataFrame) -> pl.DataFrame:
        in_col = "preprocessed_review/text"
        return data.filter(pl.col(in_col).is_not_null())


class HelpfulnessPreprocessor(Processor):
    def __init__(self):
        super().__init__("Helpfulnessratio")

    def _process_inner(self, data: pl.DataFrame) -> pl.DataFrame:
        in_col = "review/helpfulness"
        data = data.with_columns(pl.col(in_col).str.replace_all("[^\d/]", ""))
        data = data.with_columns(
            pl.col(in_col).str.split("/").list.get(0).cast(int).alias("num_helpfull"),
            pl.col(in_col).str.split("/").list.get(1).cast(int).alias("num_total"),
        )
        # Calculate the ratio
        # We set reviews where the number of total helpfulness less then 2
        # to 0
        # Set to 1 if num total is 0
        data = data.with_columns(
            pl.when(pl.col("num_total") < 2)
            .then(0)
            .otherwise(pl.col("num_helpfull") / pl.col("num_total"))
            .alias("review/helpfulness")
        )
        return data.drop(["num_helpfull", "num_total"])


class FilterUsersWithLessThen50Reviews(Processor):
    def __init__(self):
        super().__init__("Filter Users")

    def _process_inner(self, data: pl.DataFrame) -> pl.DataFrame:
        filtered_users = (
            data.select("User_id")
            .group_by(by="User_id")
            .count()
            .filter(pl.col("count") > 50)
        )
        return data.join(filtered_users, on="User_id", how="inner").drop("count")


class SplitReviewTokens(Processor):
    def __init__(self):
        super().__init__("Split Review Tokens")

    def _process_inner(self, data: pl.DataFrame) -> pl.DataFrame:
        in_col = "preprocessed_review/text"
        out_col = "preprocessed_review/tokens"
        return data.with_columns(pl.col(in_col).str.split(" ").alias(out_col))


class DropNotNeededColumns(Processor):
    def __init__(self):
        super().__init__("Drop Not Needed Columns")

    def _process_inner(self, data: pl.DataFrame) -> pl.DataFrame:
        needed_cols = [
            "Id",
            "Title",
            "User_id",
            "review/helpfulness",
            "review/score",
            "review/text",
        ]
        return data.select(needed_cols)


class DropEmptyReviewsAndUsers(Processor):
    def __init__(self):
        super().__init__("Drop Empty Reviews And Users")

    def _process_inner(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.filter(pl.col("review/text").is_not_null())


class HelpfulnessPredictionInputProcessor(Processor):
    def __init__(self):
        super().__init__("Helpfulness Prediction Input")

    def process_value(self, value):
        # Lets built an artificaial DataFrame
        data = pl.DataFrame({"review/text": [value]})
        data = self._process_inner(data)
        return data

    def _process_inner(self, data: pl.DataFrame) -> pl.DataFrame:
        output_col = "input/helpfulnessprediction"
        num_chars = data["review/text"].str.len_chars()
        num_words = data["review/text"].str.split(" ").list.len()
        num_sentences = (
            data["review/text"].str.replace_all("[!?]", ".").str.split(".").list.len()
        )
        num_long_words = (
            data["review/text"]
            .str.split(" ")
            .map_elements(lambda s: s.str.len_chars() > 6)
            .list.sum()
        )

        # Formular taken from:
        # https://www.tutorialspoint.com/readability-index-in-python-nlp
        readability_ari = (
            4.71 * (num_chars / num_words) + 0.5 * (num_words / num_sentences) - 21.43
        )
        rate_index = num_long_words / num_words

        input_values = list(
            zip(
                rate_index.cast(float),
                readability_ari.cast(float),
                num_sentences.cast(float),
                num_words.cast(float),
            )
        )
        data = data.with_columns(pl.Series(name=output_col, values=input_values))
        return data


class PreprocessingFinalizer(Processor):
    """
    This is just an artificial processor that is used to
    save the final preprocessed data to disk
    """

    def __init__(self):
        super().__init__("Finale Preprocessing")

    def _process_inner(self, data: pl.DataFrame) -> pl.DataFrame:
        return data
