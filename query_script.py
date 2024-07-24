import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models import FastText
from sklearn.neighbors import NearestNeighbors
import joblib 

# load dataset
df = pd.read_csv('processed_data.csv')

# define columns to look at
columns = ['director', 'genres', 'country', 'description', 'cast']

# store models in dictionary
models = {column: FastText.load(f'{column}_model.bin') for column in columns}

# load KNN model from pickle file
knn = joblib.load('knn_model.pkl')

def preprocess_text(text):
    return word_tokenize(text.lower())

def get_avg_embedding(tokens, model):
    embeddings = [model.wv[token] for token in tokens if token in model.wv]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.vector_size)

def combine_embeddings(tokens):
    embeddings = [get_avg_embedding(tokens, models[column]) for column in columns]
    return np.mean(embeddings, axis=0)

def query_search(query_text):
    query_tokens = preprocess_text(query_text)
    query_embedding = combine_embeddings(query_tokens)
    distance, indices = knn.kneighbors([query_embedding])
    
    results = df.iloc[indices[0]]

    # print information related to search query
    for idx, row in results.iterrows():
        print(f"Title: {row.get('title', 'N/A')}")
        print(f"Director: {row.get('director', 'N/A')}")
        print(f"Genre: {row.get('genre', 'N/A')}")
        print(f"Country: {row.get('country', 'N/A')}")
        print(f"Description: {row.get('description', 'N/A')}")
        print(f"Cast: {row.get('cast', 'N/A')}")
        print("-" * 40)

    return results


if __name__ == "__main__":
    query_text = input("Enter search query: ")
    results = query_search(query_text)
    # print("Nearest Neighbors:")
    print(results)
