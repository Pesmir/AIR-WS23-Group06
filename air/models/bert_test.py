from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import random
import numpy as np
from tqdm import tqdm


file_path = '../ReviewPreprocessing.ipc'  # Replace with your file path
data = pd.read_feather(file_path)

# Select 10 random rows from the dataset
random_reviews = data.sample(15213)

# Load Tokenizer and Models
tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
base_model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
fine_tuned_model = BertForSequenceClassification.from_pretrained('results/bert_fine-tuned_1%_model')

# Reviews and Actual Labels
reviews = random_reviews['review/text']
actual_labels = random_reviews['review/score']

# Function to get predictions
def get_predictions(model, reviews, tokenizer):
    model.eval()
    predictions = []
    with torch.no_grad():
        # Add tqdm progress bar here
        for review in tqdm(reviews, desc="Processing reviews"):
            inputs = tokenizer(review, padding=True, truncation=True, max_length=512, return_tensors='pt')
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1)
            predictions.append(float(prediction.item() + 1))
    return predictions

# Get predictions from both models
base_predictions = get_predictions(base_model, reviews, tokenizer)
fine_tuned_predictions = get_predictions(fine_tuned_model, reviews, tokenizer)

# Convert to NumPy arrays for easier calculation
actual_labels = np.array(random_reviews['review/score'].tolist())
base_predictions = np.array(base_predictions)
fine_tuned_predictions = np.array(fine_tuned_predictions)

# Evaluation metrics
correct_predictions = np.sum(actual_labels == fine_tuned_predictions)
total_predictions = len(actual_labels)
rate_of_correct_predictions = correct_predictions / total_predictions
print(f"Rate of correct predictions: {rate_of_correct_predictions}")
print(f"Absolute correct predictions: {correct_predictions} / {total_predictions}")

# Mean Absolute Error (MAE)
baseline_3_mae = np.mean(np.abs(actual_labels - 3))
baseline_rand_mae = np.mean(np.abs(actual_labels - np.random.randint(1, 6, size=total_predictions)))
prediction_mae = np.mean(np.abs(actual_labels - fine_tuned_predictions))

print(f"Baseline 3 MAE: {baseline_3_mae}")
print(f"Baseline rand MAE: {baseline_rand_mae}")
print(f"Prediction MAE: {prediction_mae}")

base_model_mae = np.mean(np.abs(actual_labels - base_predictions))

# Rate of correct predictions for the base model
base_model_correct = np.sum(actual_labels == base_predictions)
base_model_accuracy = base_model_correct / total_predictions

print(f"Base Model MAE: {base_model_mae}")
print(f"Base Model Accuracy: {base_model_accuracy}")
print(f"Base Model Absolute correct predictions: {base_model_correct} / {total_predictions}")
