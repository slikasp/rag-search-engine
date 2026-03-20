import argparse

from lib.load_data import load_movies
from lib.hybrid_search import normalize, HybridSearch

def normalize_command(scores: list):
    for s in normalize(scores):
        print(f"* {s:.4f}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="")
    normalize_parser.add_argument("scores", type=float, nargs='+', help="list of scores")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="")
    weighted_search_parser.add_argument("query", type=str, help="")
    weighted_search_parser.add_argument("--alpha", nargs='?', default=0.5, type=float, help="")
    weighted_search_parser.add_argument("--limit", nargs='?', default=5, type=int, help="")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="")
    rrf_search_parser.add_argument("query", type=str, help="")
    rrf_search_parser.add_argument("--k", nargs='?', default=60, type=int, help="")
    rrf_search_parser.add_argument("--limit", nargs='?', default=5, type=int, help="")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.scores)
        case "weighted-search":
            documents = load_movies()
            search = HybridSearch(documents)
            movies = search.weighted_search(args.query, args.alpha, args.limit)
            n = 1
            for m in movies:
                print(f"{n}. {m[1]['title']}")
                print(f"Hybrid Score: {m[1]['hybrid']:.3f}")
                print(f"BM25: {m[1]['bm25']:.3f}, Semantic: {m[1]['semantic']:.3f}")
                print(f"{m[1]['document']:.200}")
                n += 1
        case "rrf-search":
            documents = load_movies()
            search = HybridSearch(documents)
            movies = search.rrf_search(args.query, args.k, args.limit)
            n = 1
            for m in movies:
                print(f"{n}. {m[1]['title']}")
                print(f"RRF Score: {m[1]['rrf']:.3f}")
                print(f"BM25: {m[1]['bm25']}, Semantic: {m[1]['semantic']}")
                print(f"{m[1]['document']:.100}")
                n += 1
        case _:
            parser.print_help() 


if __name__ == "__main__":
    main()