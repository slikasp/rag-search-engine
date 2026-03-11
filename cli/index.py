import pickle
import math
from collections import defaultdict, Counter

from pathlib import Path

from constants import BM25_K1
from load_data import load_movies
from preprocess_strings import preprocess

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.tmap = {}
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)

    def __add_document(self, doc_id: int, text: str):
        tokens = preprocess(text)
        for t in tokens:
            if t not in self.index:
                self.index[t] = set()
            self.index[t].add(doc_id)
            self.term_frequencies[doc_id][t] += 1

    def get_documents(self, term: str):
        t = preprocess(term)
        # The t[0] part needs fixing, maybe throw an exception when there's more than 1 token?
        hits = self.index.get(t[0], set())
        return sorted(hits)
    
    def get_tf(self, doc_id: int, term: str) -> int:
        t = preprocess(term)
        if len(t) != 1:
            raise TypeError("get_tf() takes 1 term")
        return self.term_frequencies[doc_id][t[0]]
    
    def get_tokens(self, doc_id: int):
        return preprocess(f"{self.tmap[doc_id]} {self.docmap[doc_id]}")
    

    def get_bm25_idf(self, term: str) -> float:
        n = len(self.tmap)
        df = len(self.get_documents(term))
        bm25idf = math.log((n - df + 0.5) / (df + 0.5) + 1)

        return bm25idf
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1) -> float:
        tf = self.get_tf(doc_id, term)
        bm25tf = (tf * (k1 + 1)) / (tf + k1)

        return bm25tf
    
    def build(self):
        movies = load_movies()

        for m in movies:
            self.__add_document(m['id'], f"{m['title']} {m['description']}")
            self.tmap[m['id']] = m['title']
            self.docmap[m['id']] = m['description']

    def save(self):
        Path("cache").mkdir(parents=True, exist_ok=True)
        with open("cache/index.pkl", 'wb') as f_index:
            pickle.dump(self.index, f_index)
        with open("cache/tmap.pkl", 'wb') as f_tmap:
            pickle.dump(self.tmap, f_tmap)
        with open("cache/docmap.pkl", 'wb') as f_docmap:
            pickle.dump(self.docmap, f_docmap)
        with open("cache/term_frequencies.pkl", 'wb') as f_term_frequencies:
            pickle.dump(self.term_frequencies, f_term_frequencies)
        

    def load(self):
        try:
            with open("cache/index.pkl", 'rb') as f_index:
                self.index = pickle.load(f_index)
            with open("cache/tmap.pkl", 'rb') as f_tmap:
                self.tmap = pickle.load(f_tmap)
            with open("cache/docmap.pkl", 'rb') as f_docmap:
                self.docmap = pickle.load(f_docmap)
            with open("cache/term_frequencies.pkl", 'rb') as f_term_frequencies:
                self.term_frequencies = pickle.load(f_term_frequencies)
        except FileNotFoundError:
            raise FileNotFoundError("Cache files do not exist. Build the index first.")