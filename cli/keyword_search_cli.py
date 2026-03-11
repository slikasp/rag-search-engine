#!/usr/bin/env python3

import argparse
import math

from constants import BM25_K1
from index import InvertedIndex


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

    tfidf_parser = subparsers.add_parser("tfidf", help="TF-IDF")
    tfidf_parser.add_argument("id", type=int, help="id to check")
    tfidf_parser.add_argument("term", type=str, help="term to check")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")

    token_parser = subparsers.add_parser("test_token", help="test stems")
    token_parser.add_argument("id", type=int, help="id to check")

    args = parser.parse_args()

    movies = InvertedIndex()

    # TODO: put all logic to separate command functions
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
            tf = movies.get_tf(args.id, args.term)
            print(f"Term '{args.term}' appears {tf} times in '{args.id}' ({movies.tmap[args.id]}) ")
        case "idf":
            try:
                movies.load()
            except Exception as e:
                print(e)
            total_doc_count = len(movies.tmap)
            term_match_doc_count = len(movies.get_documents(args.term))
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            try:
                movies.load()
            except Exception as e:
                print(e)
            tf_idf = calc_tfidf(movies, args.id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.id}': {tf_idf:.2f}")
        case "bm25idf":
            try:
                movies.load()
            except Exception as e:
                print(e)
            bm25idf = movies.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            try:
                movies.load()
            except Exception as e:
                print(e)
            bm25tf = movies.get_bm25_tf(args.id, args.term, args.k1)
            print(f"BM25 TF score of '{args.term}' in document '{args.id}': {bm25tf:.2f}")
        case "test_token":
            movies.load()
            print(movies.get_tokens(args.id))
        case _:
            parser.print_help()



if __name__ == "__main__":
    main()