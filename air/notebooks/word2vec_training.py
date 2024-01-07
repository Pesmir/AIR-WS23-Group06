import logging

import numpy as np
from gensim.models import Word2Vec
import nltk

from air.data.load import load_preprocessed_data

nltk.download('punkt')

# Use this to log information for build_vocab and train
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

input_df = load_preprocessed_data()
print(input_df.columns)
sentences = input_df['preprocessed_review/tokens'].to_list()
print(sentences[0])

# Create model
model = Word2Vec(min_count=20,
                 window=2,
                 sample=6e-5,
                 alpha=0.03,
                 min_alpha=0.0007,
                 negative=20,
                 workers=4)
# Add words
model.build_vocab(corpus_iterable=sentences, progress_per=100)
model.train(sentences, total_examples=model.corpus_count, epochs=10, report_delay=1.0)
print("Training done")

# Create document vectors by averaging the word vectors
document_vectors = []
count = 0
max_vector_length = 0

for sentence in sentences:
    count += 1
    if count % 100 == 0:
        print(count)

    word_vectors = []
    for word in sentence:

        if word in model.wv.index_to_key:
            word_vectors.append(model.wv[word])
            vector_length = len(model.wv[word])
            if vector_length > max_vector_length:
                max_vector_length = vector_length

    if word_vectors:
        avg_vector = np.mean(word_vectors, axis=0)
        document_vectors.append(avg_vector)
    else:
        document_vectors.append(None)


print("document_vectors done")
# Pad vectors to a fixed length
document_vectors_padded = np.array([np.pad(vector, (0, max_vector_length - len(vector))) for vector in document_vectors])
# Save for usage with other models later
np.save('air/data/word2vec_document_vectors.npy', document_vectors_padded)
