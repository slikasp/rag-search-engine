#!/usr/bin/env python3

import argparse
import math

from lib.constants import BM25_K1, BM25_B
from lib.commands import search_command, build_command, tf_command, idf_command, tfidf_command, bm25idf_command, mb25tf_command, bm25search_command
from lib.keyword_search import InvertedIndex


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
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("limit", type=int, nargs='?', default=5, help="Number of titles to return")

    token_parser = subparsers.add_parser("test_token", help="test stems")
    token_parser.add_argument("id", type=int, help="id to check")

    args = parser.parse_args()

    match args.command:
        case "search":
            search_command(args.query)
        case "build":
            build_command()
        case "tf":
            tf_command(args.id, args.term)
        case "idf":
            idf_command(args.term)
        case "tfidf":
            tfidf_command(args.id, args.term)
        case "bm25idf":
            bm25idf_command(args.term)
        case "bm25tf":
            mb25tf_command(args.id, args.term, args.k1, args.b)
        case "bm25search":
            bm25search_command(args.query, args.limit)
        case "test_token":
            movies = InvertedIndex()
            movies.load()
            print(movies.__get_tokens(args.id))
        case _:
            parser.print_help()



if __name__ == "__main__":
    main()