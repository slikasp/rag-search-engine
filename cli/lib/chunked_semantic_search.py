import re
import numpy as np
import json
import os

from lib.semantic_search import SemanticSearch

def semantic_chunk(text: str, size: int, overlap: int) -> list[str]:
        if overlap < 0:
            overlap = 0
        
        text = text.strip()
        if len(text) == 0:
            return []

        pattern = r"(?<=[.!?])\s+"
        sentences = re.split(pattern, text)
        cleaned_sentences = []
        for s in sentences:
            c_s = s.strip()
            if len(c_s) != 0:
                cleaned_sentences.append(c_s)

        chunks = []
        n = 0
        while n < (len(cleaned_sentences) - overlap):
            chunk = cleaned_sentences[n:n+size]
            chunks.append(" ".join(chunk))
            n += size - overlap
        return chunks

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def generate_embedding(self, text: str):
        if not text.strip():
             raise ValueError(f"Empty string provided: {text}")
        embedding = self.model.encode([text])
        return embedding[0]

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        chunks = []
        chunk_metadata = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            if doc == "":
                continue
            doc_chunks = semantic_chunk(doc['description'], 4, 1)
            chunk_id = 0
            for c in doc_chunks:
                chunk_metadata.append({
                    'movie_idx': doc['id'],
                    'chunk_idx': chunk_id,
                    'total_chunks': len(doc_chunks)
                })
                chunks.append(c)
                chunk_id += 1

        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        with open("cache/chunk_embeddings.npy", 'wb') as f_embeddings:
            np.save(f_embeddings, self.chunk_embeddings)

        with open("cache/chunk_metadata.json", 'w') as f_metadata:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(chunks)}, f_metadata, indent=2)

        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc

        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):
            with open("cache/chunk_embeddings.npy", 'rb') as f_embeddings:
                self.chunk_embeddings = np.load(f_embeddings)
            with open("cache/chunk_metadata.json", 'r') as f_metadata:
                metadata = json.load(f_metadata)
                self.chunk_metadata = metadata['chunks']
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        q_em = self.generate_embedding(query)
        chunk_scores = []
        if self.chunk_embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_chunk_embeddings` first.")
        if self.chunk_metadata is None:
            raise ValueError("No metadata loaded. Call `load_or_create_chunk_embeddings` first.")
        for i, c_em in enumerate(self.chunk_embeddings):
            metadata = self.chunk_metadata[i]
            movie_idx = metadata["movie_idx"]
            chunk_idx = metadata["chunk_idx"]
            chunk_scores.append({
                'chunk_idx': chunk_idx,
                'movie_idx': movie_idx,
                'score': cosine_similarity(q_em ,c_em)
            })

        movie_scores = {}
        for c_s in chunk_scores:
            if c_s['movie_idx'] not in movie_scores:
                movie_scores[c_s['movie_idx']] = c_s['score']
            elif movie_scores[c_s['movie_idx']] < c_s['score']:
                movie_scores[c_s['movie_idx']] = c_s['score']
        
        top_scores = sorted(movie_scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        results = []
        for m_id, m_score in top_scores:
            results.append({
                "id": m_id,
                "title": self.document_map[m_id]['title'],
                "document": self.document_map[m_id]['description'][:100],
                "score": round(m_score, 4),
                "metadata": metadata or {}
            })
        
        return results