import numpy as np

from lib.load_data import load_movies
from PIL import Image
from sentence_transformers import SentenceTransformer

def image_search_command(image_path: str) -> list:
    documents = load_movies()
    search = MultimodalSearch(documents)
    results = search.search_with_image(image_path)
    return results

def verify_image_embedding(image_path: str):
    search = MultimodalSearch([])
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

class MultimodalSearch:
    def __init__(self, documents: list, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            embeddings = self.model.encode([img])
        return embeddings[0]

    def search_with_image(self, image_path: str):
        embedding = self.embed_image(image_path)
        cos_sim = {}
        for i, t_e in enumerate(self.text_embeddings):
            cos_sim[int(self.documents[i]['id'])-1] = cosine_similarity(t_e, embedding)
        cos_sim = sorted(
            cos_sim.items(),
            key=lambda item: item[1],
            reverse=True
        )[:5]
        results = []
        for k, v in cos_sim:
            results.append({
                'id': k,
                'title': self.documents[k]['title'],
                'description': self.documents[k]['description'],
                'similarity': v,
            })
        return results