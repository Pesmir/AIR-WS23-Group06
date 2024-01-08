import logging

import numpy as np
from gensim.models import Word2Vec
import nltk
from polars.dependencies import pickle
import polars as pl

from air.data.load import load_preprocessed_data

nltk.download('punkt')

# Use this to log information for build_vocab and train
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


def train_word2vec():
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
    model.save("air/data/models/word2vec.model")
    print("Training done")


def create_vectors():
    input_df = load_preprocessed_data()
    sentences = input_df['preprocessed_review/tokens'].to_list()
    model = Word2Vec.load("air/data/models/word2vec.model")
    # Create document vectors by averaging the word vectors
    document_vectors = []
    count = 0
    max_vector_length = 0

    for sentence in sentences:
        count += 1
        if count % 10000 == 0:
            print(count)
            with open('air/data/word2vec_document_vectors_{}.pkl'.format(count), "wb") as f:
                pickle.dump(document_vectors, f)

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

    with open('air/data/word2vec_document_vectors_{}.pkl'.format(count), "wb") as f:
        pickle.dump(document_vectors, f)

def pad_vectors():
    # Pad vectors to a fixed length
    document_vectors_padded = []
    rows_to_delete = []
    with open("air/data/word2vec_document_vectors_903090.pkl", "rb") as f:
        document_vectors = pickle.load(f)

    max_vector_length = max(len(vector) for vector in document_vectors if vector is not None)
    print("max_vector_length: ")
    print(max_vector_length)

    count = 0
    for vector in document_vectors:
        count += 1
        if count % 10000 == 0:
            print(count)
        if vector is not None:
            padding = max_vector_length - len(vector)
            padded_vector = np.pad(vector, (0, padding))
            document_vectors_padded.append(padded_vector)
        else:
            rows_to_delete.append(count)

    with open('air/data/word2vec_document_vectors_padded.pkl', "wb") as f:
        pickle.dump(document_vectors_padded, f)
    with open('air/data/rows_to_delete.pkl', "wb") as f:
        pickle.dump(rows_to_delete, f)

def create_new_dataset():
    data = load_preprocessed_data()
    with open("air/data/word2vec_document_vectors_padded.pkl", "rb") as f:
        document_vectors = pickle.load(f)
    with open("air/data/rows_to_delete.pkl", "rb") as f:
        rows_to_delete = pickle.load(f)
    new_column_name = "review/vectors"
    old_index = 0
    rows_to_delete.append(len(data))
    df_new = data.clear()
    for row in rows_to_delete:
        rows = data[old_index:row]
        df_new = pl.concat([df_new, rows])
        old_index = row + 1
    series = pl.Series(name=new_column_name, values=document_vectors)
    res = df_new.with_columns(series)
    with open('air/data/dataset_with_vectors.pkl', "wb") as f:
        pickle.dump(res, f)

#pad_vectors()
#create_new_dataset()