import argparse
from lib.load_data import load_movies, load_golden
from lib.hybrid_search import HybridSearch

def retrieved_relevant(relevant_titles: list, retrieved_titles: list) -> int:
    relevant = 0
    for m in retrieved_titles:
        if m in relevant_titles:
            relevant += 1
    return relevant

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument("--limit",type=int, default=5, help="Number of results to evaluate (k for precision@k, recall@k)")

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    test_cases = load_golden()
    
    documents = load_movies()
    search = HybridSearch(documents)

    for case in test_cases:

        movies = search.rrf_search(case['query'], 60, limit)
        relevant_titles = case['relevant_docs']
        retrieved_titles = [m[1]['title'] for m in movies]
        rr = retrieved_relevant(relevant_titles, retrieved_titles)
        precision = rr / len(retrieved_titles)
        recall = rr / len(relevant_titles)
        f1 = 2 * (precision * recall) / (precision + recall)

        print(f"- Query: {case['query']}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        # print(f"  - Retrieved: {retrieved_titles}")
        # print(f"  - Relevant: {relevant_titles}")



if __name__ == "__main__":
    main()