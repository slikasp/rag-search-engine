import numpy as np
import os

from lib.load_data import load_movies
from sentence_transformers import SentenceTransformer

def verify_model():
        search = SemanticSearch()
        print(f"Model loaded: {search.model.__str__()}")
        print(f"Max sequence length: {search.model.max_seq_length}")

def embed_text(text: str):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    search = SemanticSearch()
    documents = load_movies()
    embeddings = search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query: str):
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def build_embeddings(self, documents):
        self.documents = documents
        movies = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            movies.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movies, show_progress_bar=True)
        with open("cache/movie_embeddings.npy", 'wb') as f_embeddings:
            np.save(f_embeddings, self.embeddings)
        return self.embeddings
        
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    # text - expects one word string
    def generate_embedding(self, text: str):
        if not text.strip():
             raise ValueError(f"Empty string provided: {text}")
        embedding = self.model.encode([text])
        return embedding[0]
        
    def search(self, query: str, limit: int=5) -> list[dict]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        if self.documents is None:
            raise ValueError("No documents loaded. Call `load_or_create_embeddings` first.")
        q_em = self.generate_embedding(query)
        cos_sim = []
        for i, d_em in enumerate(self.embeddings):
            cos_sim.append((cosine_similarity(q_em ,d_em), self.documents[i]))
        cos_sim.sort(key=lambda x: x[0], reverse=True)
        results = []
        for cs in cos_sim[:limit]:
            results.append({
                'score': cs[0],
                'title': cs[1]['title'],
                'description': cs[1]['description'],
            })
        return results