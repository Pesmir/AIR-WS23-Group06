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

    def _process_inner(self, data: pl.DataFrame) -> pl.DataFrame:
        out_col = "preprocessed_review/text"
        data = data.with_columns(pl.col("review/text").alias(out_col))

        # Split data into 5 chunks
        start_time = time.time()
        data_chunks = []
        num_processes = 10
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
        data = data.with_columns(pl.col(in_col).str.split("/"))
        data = data.with_columns(
            pl.col(in_col).map_elements(
                lambda x: (int(x[0]) / int(x[1]) if int(x[1]) != 0 else 0)
            )
        )
        return data
