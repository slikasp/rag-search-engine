import os

from .keyword_search import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch

def normalize(scores: list) -> list:
    n_scores = len(scores)
    if n_scores == 0:
        return []
    
    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0 * n_scores]
    else:
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
def rank(scores: dict) -> dict:
    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    n = 1
    for k in sorted_scores.keys():
        sorted_scores[k] = n
        n += 1
    return sorted_scores

def hybrid_score(bm25_score, semantic_score, alpha=0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank, k=60):
    return 1 / (k + rank)

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25 = self._bm25_search(query, limit*500)
        bm25_ids = [t[0] for t in bm25]
        bm25_vals = [t[1] for t in bm25]
        bm25_norm = dict(zip(bm25_ids, normalize(bm25_vals)))
        css = self.semantic_search.search_chunks(query, limit*500)
        css_ids = [d['id'] for d in css]
        css_vals = [d['score'] for d in css]
        css_norm = dict(zip(css_ids, normalize(css_vals)))
        scores = {}
        # 1 - create scores instances for bm25 and semantic results
        # print(">>>BM25: ", bm25_ids)
        # print(">>>Semantic: ", css_ids)
        for id in set(bm25_ids + css_ids):
            scores[id] = {
                'title': self.documents[id-1]['title'],
                'document': self.documents[id-1]['description'],
                'bm25': 0.0,
                'semantic': 0.0,
                'hybrid': 0.0
            }
        # 2 - update bm25 and semantic numbers with normalized scores
        for id, v in bm25_norm.items():
            scores[id]['bm25'] = v
        for id, v in css_norm.items():
            scores[id]['semantic'] = v
        # 3 - calculate hybrid
        for v in scores.values():
            v['hybrid'] = hybrid_score(v['bm25'], v['semantic'], alpha)
        # 4 - sort and filter
        top = sorted(
            scores.items(),
            key=lambda item: item[1]['hybrid'],
            reverse=True
        )[:limit]
        return top

    def rrf_search(self, query, k, limit=10):
        bm25 = self._bm25_search(query, limit*500)
        bm25_dict = dict(bm25)
        bm25_ranked = rank(bm25_dict)
        css = self.semantic_search.search_chunks(query, limit*500)
        css_dict = {d['id']: d['score'] for d in css}
        css_ranked = rank(css_dict)
        ranks = {}
        for id in set([t[0] for t in bm25] + [d['id'] for d in css]):
            ranks[id] = {
                'title': self.documents[id-1]['title'],
                'document': self.documents[id-1]['description'],
                'bm25': 0,
                'semantic': 0,
                'rrf': 0.0
            }
        for id, v in bm25_ranked.items():
            ranks[id]['bm25'] = v
            ranks[id]['rrf'] += rrf_score(v, k)
        for id, v in css_ranked.items():
            ranks[id]['semantic'] = v
            ranks[id]['rrf'] += rrf_score(v, k)
        top = sorted(
            ranks.items(),
            key=lambda item: item[1]['rrf'],
            reverse=True
        )[:limit]
        return top