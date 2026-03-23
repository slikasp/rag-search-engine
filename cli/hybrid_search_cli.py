import argparse

from lib.load_data import load_movies
from sentence_transformers import CrossEncoder
from test_gemini import enhance_spelling, enhance_rewrite, enhance_expand, rerank_individual, rerank_batch, evaluate
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
    rrf_search_parser.add_argument("--evaluate", action='store_true', help="")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="")

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
            # check options
            match args.enhance:
                case "spell":
                    query = enhance_spelling(args.query)
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")
                case "rewrite":
                    query = enhance_rewrite(args.query)
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")
                case "expand":
                    query = enhance_expand(args.query)
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")
                case _:
                    query = args.query
            rerank = False
            match args.rerank_method:
                case "individual":
                    limit = args.limit * 5
                    rerank = True
                case "batch":
                    limit = args.limit * 5
                    rerank = True
                case "cross_encoder":
                    limit = args.limit * 5
                    rerank = True
                case _:
                    limit = args.limit
            # run search
            documents = load_movies()
            search = HybridSearch(documents)
            movies = search.rrf_search(query, args.k, limit)
            # postprocess based on options
            if rerank:
                print(f"Re-ranking top {args.limit} results using {args.rerank_method} method...")
                print(f"Reciprocal Rank Fusion Results for '{args.query}' (k={args.k})")
                if args.rerank_method == "individual":
                    for m in movies:
                        rr_score = rerank_individual(query, m[1]['title'], m[1]['document'])
                        m[1]['rrs'] = rr_score
                    movies = sorted(
                        movies,
                        key=lambda item: item[1]['rrs'],
                        reverse=True
                    )[:args.limit]
                elif args.rerank_method == "batch":
                    rr_ids = rerank_batch(query, movies)
                    n = 1
                    for m in movies:
                        m[1]['rrr'] = rr_ids.index(m[0]) + 1
                    movies = sorted(
                        movies,
                        key=lambda item: item[1]['rrr'],
                        reverse=False
                    )[:args.limit]
                elif args.rerank_method == "cross_encoder":
                    pairs = []
                    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                    for m in movies:
                        pairs.append([query, f"{m[1]['title']} - {m[1]['document']}"])
                    # `predict` returns a list of numbers, one for each pair
                    scores = cross_encoder.predict(pairs)
                    n = 0
                    for m in movies:
                        m[1]['ces'] = scores[n]
                        n += 1
                    movies = sorted(
                        movies,
                        key=lambda item: item[1]['ces'],
                        reverse=True
                    )[:args.limit]
                   
            # output
            n = 1
            for m in movies:
                print(f"{n}. {m[1]['title']}")
                if args.rerank_method == "individual":
                    print(f"Re-rank Score: {m[1]['rrs']}/10")
                if args.rerank_method == "batch":
                    print(f"Re-rank Rank: {m[1]['rrr']}")
                if args.rerank_method == "cross_encoder":
                    print(f" Cross Encoder Score: {m[1]['ces']:.3f}")
                print(f"RRF Score: {m[1]['rrf']:.3f}")
                print(f"BM25: {m[1]['bm25']}, Semantic: {m[1]['semantic']}")
                # print(f"{m[1]['document']:.100}")
                n += 1
            
            if args.evaluate:
                print("AI Evaluation:")
                eval = evaluate(query, [m[1]['title']+":"+m[1]['document'] for m in movies])
                n = 1
                for m in movies:
                    print(f"{n}. {m[1]['title']}: {eval[n-1]}/3")

        case _:
            parser.print_help() 


if __name__ == "__main__":
    main()