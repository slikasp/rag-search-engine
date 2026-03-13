from sentence_transformers import SentenceTransformer

def verify_model():
        search = SemanticSearch()
        print(f"Model loaded: {search.model.__str__()}")
        print(f"Max sequence length: {search.model.max_seq_length}")

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    