import os
import torch
import polars as pl
import xgboost as xgb
import numpy as np
import pickle
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from air.processing.processor import Processor
from air.processing.preprocessing import (
    HelpfulnessPredictionInputProcessor,
    ReviewPreprocessor,
)


class ModelPredictionProcessorBase(Processor):
    def __init__(self, name):
        self.name = name
        self._check_point_path = f"./air/data/checkpoints/{name}.ipc"

    def _predict_inner(self, data: pl.DataFrame, output_col: str) -> pl.DataFrame:
        raise NotImplementedError()

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        print(f"Processing {self.name}...")
        if os.path.exists(self._check_point_path):
            print(f"Found checkpoint for {self.name}, loading...")
            result = pl.read_ipc(self._check_point_path)
        else:
            result = self._predict_inner(data, self.name)
            print(f"Saving checkpoint for {self.name} in {self._check_point_path}...")
            result.write_ipc(self._check_point_path)
        return result

    def predict_value(self, value) -> pl.DataFrame:
        value = ReviewPreprocessor().process_value(value)
        return self._predict_inner(value, self.name)[self.name].to_numpy().flatten()[0]


class XGBoostModelPredictionProcessor(ModelPredictionProcessorBase):
    def __init__(self):
        super().__init__("xgboost")
        with open("air/data/models/xgboost_full.pkl", "rb") as f:
            self._model = pickle.load(f)
        with open("air/data/models/xgboost_countvectorizer_full.pkl", "rb") as f:
            self._vectorizer = pickle.load(f)

    def _predict_inner(self, data: pl.DataFrame, output_col: str) -> pl.DataFrame:
        x = self._vectorizer.transform(data["preprocessed_review/text"])
        x = xgb.DMatrix(x, enable_categorical=True)
        predictions = self._model.predict(x) + 1
        data = data.with_columns(pl.Series(name=output_col, values=predictions))
        print(data)
        return data


class BertModelPredictionProcessor(ModelPredictionProcessorBase):
    def __init__(self, fine_tuned=False):
        name = "bert_fine-tuned" if fine_tuned else "bert-base"
        super().__init__(name)
        self._tokenizer = BertTokenizer.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment"
        )
        if fine_tuned:
            self._model = BertForSequenceClassification.from_pretrained(
                "results/bert_fine-tuned_1%_model"
            )
        else:
            self._model = BertForSequenceClassification.from_pretrained(
                "nlptown/bert-base-multilingual-uncased-sentiment"
            )

    def _predict_inner(self, data: pl.DataFrame, output_col: str) -> pl.DataFrame:
        self._model.eval()
        predictions = []

        reviews = data["preprocessed_review/text"]
        with torch.no_grad():
            for review in tqdm(reviews, desc="Processing reviews"):
                inputs = self._tokenizer(
                    review,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                outputs = self._model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=1)
                predictions.append(float(prediction.item() + 1))
        data = data.with_columns(pl.Series(name=output_col, values=predictions))
        return data


class XGBoostHelpfulnessPredictionProcessor(ModelPredictionProcessorBase):
    def __init__(self):
        super().__init__("xgboost_helpfulness")
        with open("air/data/models/xgboost_helpfulness.pkl", "rb") as f:
            self._model = pickle.load(f)

    def predict_value(self, value) -> pl.DataFrame:
        value = HelpfulnessPredictionInputProcessor().process_value(value)
        return self._predict_inner(value, self.name)[self.name].to_numpy().flatten()[0]

    def _predict_inner(self, data: pl.DataFrame, output_col: str) -> pl.DataFrame:
        x = np.stack(data["input/helpfulnessprediction"].to_numpy())
        x = xgb.DMatrix(x)
        predictions = np.clip(self._model.predict(x), 0.0, 1.0)
        data = data.with_columns(pl.Series(name=output_col, values=predictions))
        return data
