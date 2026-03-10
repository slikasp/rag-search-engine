#!/usr/bin/env python3

import argparse
import math

from preprocess_strings import test_stem
from index import InvertedIndex


def title_search(movies: InvertedIndex, query: str) -> list[int]:
    limit = 5
    matches = []
    for q in query.split():
        matches.extend(movies.get_documents(q))
        if len(matches) >= limit:
            return matches[:limit]
    return matches[:limit]



def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build search index")

    tf_parser = subparsers.add_parser("tf", help="Check term frequency")
    tf_parser.add_argument("id", type=int, help="document ID")
    tf_parser.add_argument("term", type=str, help="term to check")

    idf_parser = subparsers.add_parser("idf", help="Inverse Document Frequency")
    idf_parser.add_argument("term", type=str, help="term to check")

    token_parser = subparsers.add_parser("token", help="test stems")
    token_parser.add_argument("id", type=int, help="term to check")

    args = parser.parse_args()

    movies = InvertedIndex()

    match args.command:
        case "search":
            try:
                movies.load()
            except Exception as e:
                print(e)
            print(f"Index loaded. Searching for: {args.query}")
            results = title_search(movies, args.query)
            for id in results:
                print(f"{id}. {movies.tmap[id]}")
        case "build":
            print('Building...')
            movies.build()
            movies.save()
            print('Build Complete!')
        case "tf":
            try:
                movies.load()
            except Exception as e:
                print(e)
            f = movies.get_tf(args.id, args.term)
            print(f"Term '{args.term}' appears {f} times in '{args.id}' ({movies.tmap[args.id]}) ")
        case "idf":
            try:
                movies.load()
            except Exception as e:
                print(e)
            total_doc_count = len(movies.tmap)
            term_match_doc_count = len(movies.get_documents(args.term))
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "token":
            movies.load()
            print(movies.get_tokens(args.id))
        case _:
            parser.print_help()



if __name__ == "__main__":
    main()