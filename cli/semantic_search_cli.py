#!/usr/bin/env python3

import argparse


from lib.load_data import load_movies
from lib.semantic_search import SemanticSearch, verify_model, embed_text, verify_embeddings, embed_query_text

def main():    
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("verify", help="Verify the semantic search model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Check embeddings for text input")
    embed_text_parser.add_argument("text", type=str, help="Text to check")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify loading of embeddings")

    embedquery_parser = subparsers.add_parser("embedquery", help="Check embeddings for text input")
    embedquery_parser.add_argument("query", type=str, help="Text to check")

    search_parser = subparsers.add_parser("search", help="Search for movies using semantic search")
    search_parser.add_argument("query", type=str, help="Movie to search for")
    search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Number of titles to return")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, nargs='?', default=200, help="Chunk character limit")
    chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="How much overlap between chunks")


    args = parser.parse_args()

    def chunk_command(text: str, size: int, overlap: int) -> list[str]:
        if overlap < 0:
            overlap = 0
        words = text.split()
        chunks = []
        n = 0
        while n < (len(words) - overlap):
            chunk = words[n:n+size]
            chunks.append(" ".join(chunk))
            n += size - overlap
        return chunks

    match args.command:
        case "search":
            search = SemanticSearch()
            documents = load_movies()
            search.load_or_create_embeddings(documents)
            movies = search.search(args.query, args.limit)
            n = 1
            for m in movies:
                print(f"{n}. {m['title']} (score: {m['score']:.4f})")
                print(f"   {m['description']:.100}")
                n += 1
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "chunk":
            chunks = chunk_command(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters")
            n = 1
            for c in chunks:
                print(f"{n}. {c}")
                n += 1
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()