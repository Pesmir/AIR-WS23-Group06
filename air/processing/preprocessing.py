import polars as pl
import nltk
import string
from functools import lru_cache

from air.processing.processor import Processor


stem_cnt = 0
totla_cnt = 0


class CachedPorterStemmer(nltk.PorterStemmer):
    @lru_cache(maxsize=200_000)
    def stem(self, word):
        return super().stem(word)


class PreProcessor(Processor):
    def __init__(self):
        self.to_remove = set(
            nltk.corpus.stopwords.words("english") + list(string.punctuation)
        )
        self.stemmer = CachedPorterStemmer()
        super().__init__("Review Preprocessing")

    def to_lower(self, data, col):
        print("     Converting strings to lowercase...")
        return data.with_columns(pl.col(col).str.to_lowercase())

    def create_tokens(self, data, col):
        print("     Creating tokens...")
        return data.with_columns(pl.col(col).str.split(" "))

    def remove_punctuation(self, data, col):
        print("     Removing punctuation...")
        return data.with_columns(pl.col(col).str.replace_all("[^\w\s]", ""))

    def remove_numbers(self, data, col):
        print("     Removing numbers...")
        return data.with_columns(pl.col(col).str.replace_all("\d+", ""))

    def _stem_words(self, word_tokens):
        global stem_cnt
        global total_cnt
        stem_cnt += 1
        done_perc = (stem_cnt / total_cnt) * 100
        print(f"     Stemming words... ({done_perc:.2f}%)", end="\r")

        for idx, word in enumerate(word_tokens):
            word_tokens[idx] = self.stemmer.stem(word)
        return word_tokens

    def stem_words(self, data, col):
        print("     Stemming words...")
        return data.with_columns(pl.col(col).map_elements(self._stem_words))

    def join_tokens(self, data, col):
        print("     Joining tokens...")
        return data.with_columns(pl.col(col).list.join(" "))

    def _process_inner(self, data: pl.DataFrame) -> pl.DataFrame:
        global total_cnt
        # slice data to only include the first 1000 rows
        data = data.slice(0, 1000)
        total_cnt = data.select(pl.count())[0, 0]
        out_col = "preprocessed_review/text"
        data = data.with_columns(pl.col("review/text").alias(out_col))

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
