#!/usr/bin/env python3

import argparse

from load_data import load_movies
from preprocess_strings import preprocess


def title_search(movies: dict, query: str) -> list[str]:
    query_tokens = preprocess(query)

    results = []
    counter = 0

    for movie in movies:
        movie_tokens = preprocess(movie['title'])
        if any(qt in tt for qt in query_tokens for tt in movie_tokens):
            results.append(movie['title'])
            counter += 1
        if counter >= 5:
            break

    return results



def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            movies = load_movies()
            titles = title_search(movies, args.query)
            n = 0
            for title in titles:
                n += 1
                print(f"{n}. {title}")
        case _:
            parser.print_help()



if __name__ == "__main__":
    main()