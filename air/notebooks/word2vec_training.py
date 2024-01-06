# import logging

import numpy as np
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize

from air.data.load import load_preprocessed_data_small

nltk.download('punkt')

# Use this to log information for build_vocab and train
# logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

input_df = load_preprocessed_data_small()
# Drop tables with empty review text
input_df = input_df.filter(input_df['preprocessed_review/text'].is_not_null())
sentences = input_df['preprocessed_review/text'].to_list()
# Tokenize TODO: add to preprocessing
tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

# Create model
model = Word2Vec(min_count=20,
                 window=2,
                 sample=6e-5,
                 alpha=0.03,
                 min_alpha=0.0007,
                 negative=20,
                 workers=4)
# Add words
model.build_vocab(corpus_iterable=tokenized_sentences, progress_per=100)
model.train(sentences, total_examples=model.corpus_count, epochs=10, report_delay=1.0)

max_vector_length = max(len(model.wv[word]) for sentence in tokenized_sentences for word in sentence if word in model.wv.index_to_key)

# Filter out sentences with no words in the vocabulary
valid_sentences = [sentence for sentence in tokenized_sentences if any(word in model.wv.index_to_key for word in sentence)]

# Create document vectors by averaging the word vectors
document_vectors = []
for sentence in valid_sentences:
    word_vectors = [model.wv[word] for word in sentence if word in model.wv.index_to_key]
    if word_vectors:
        avg_vector = np.mean(word_vectors, axis=0)
        document_vectors.append(avg_vector)

# Pad vectors to a fixed length
document_vectors_padded = np.array([np.pad(vector, (0, max_vector_length - len(vector))) for vector in document_vectors])
# Save for usage with other models later
np.save('air/data/word2vec_document_vectors.npy', document_vectors_padded)
