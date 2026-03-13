import math

from lib.constants import BM25_K1, BM25_B
from lib.keyword_search import InvertedIndex

def load_index() -> InvertedIndex:
    index = InvertedIndex()
    try:
        index.load()
    except Exception as e:
        print(e)

    return index

def title_search(movies: InvertedIndex, query: str) -> list[int]:
    limit = 5
    matches = []
    for q in query.split():
        matches.extend(movies.get_documents(q))
        if len(matches) >= limit:
            return matches[:limit]
    return matches[:limit]

def calc_tfidf(movies: InvertedIndex, id: int, query: str) -> float:
    tf = movies.get_tf(id, query)
    total_doc_count = len(movies.tmap)
    term_match_doc_count = len(movies.get_documents(query))
    idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    return tf*idf

def search_command(query: str):
    index = load_index()
    print(f"Index loaded. Searching for: {query}")
    results = title_search(index, query)
    for id in results:
        print(f"{id}. {index.tmap[id]}")

def build_command():
    index = InvertedIndex()
    print('Building...')
    index.build()
    index.save()
    print('Build Complete!')


def tf_command(id: int, term: str):
    index = load_index()
    tf = index.get_tf(id, term)
    print(f"Term '{term}' appears {tf} times in '{id}' ({index.tmap[id]}) ")

def idf_command(term: str):
    index = load_index()
    total_doc_count = len(index.tmap)
    term_match_doc_count = len(index.get_documents(term))
    idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
    print(f"Inverse document frequency of '{term}': {idf:.2f}")

def tfidf_command(id: int, term: str):
    index = load_index()
    tf_idf = calc_tfidf(index, id, term)
    print(f"TF-IDF score of '{term}' in document '{id}': {tf_idf:.2f}")

def bm25idf_command(term: str):
    index = load_index()
    bm25idf = index.get_bm25_idf(term)
    print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")

def mb25tf_command(id: int, term: str, k1: float=BM25_K1, b: float=BM25_B):
    index = load_index()
    bm25tf = index.get_bm25_tf(id, term, k1, b)
    print(f"BM25 TF score of '{term}' in document '{id}': {bm25tf:.2f}")

def bm25search_command(query: str, limit: int=5):
    index = load_index()
    results = index.bm25_search(query, limit)
    n = 1
    for r in results:
        print(f"{n}. {index.tmap[r[0]]} - Score: {r[1]:.2f}")
        n += 1