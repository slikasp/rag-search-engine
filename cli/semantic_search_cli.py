#!/usr/bin/env python3

import argparse
import re

from lib.load_data import load_movies
from lib.semantic_search import SemanticSearch, verify_model, embed_text, verify_embeddings, embed_query_text
from lib.chunked_semantic_search import semantic_chunk, ChunkedSemanticSearch

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

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk text semantically")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs='?', default=4, help="Chunk character limit")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="How much overlap between chunks")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search for movies using semantic chunked search")
    search_chunked_parser.add_argument("query", type=str, help="Movie to search for")
    search_chunked_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Number of titles to return")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Embed chunks")

    test_parser = subparsers.add_parser("test", help="test")

    args = parser.parse_args()

    def search_command():
        search = SemanticSearch()
        documents = load_movies()
        search.load_or_create_embeddings(documents)
        movies = search.search(args.query, args.limit)
        n = 1
        for m in movies:
            print(f"{n}. {m['title']} (score: {m['score']:.4f})")
            print(f"   {m['description']:.100}")
            n += 1  
    
    def search_chunked_command():
        search = ChunkedSemanticSearch()
        documents = load_movies()
        search.load_or_create_chunk_embeddings(documents)
        movies = search.search_chunks(args.query, args.limit)
        n = 1
        for m in movies:
            print(f"\n{n}. {m['title']} (score: {m['score']:.4f})")
            print(f"   {m['document']}...")
            n += 1

    def test_command():
        # search = ChunkedSemanticSearch()
        # documents = load_movies()
        # search.load_or_create_chunk_embeddings(documents)
        # print("Embeddings: ", search.chunk_embeddings.size)
        # print("Metadata: ", len(search.chunk_metadata))
        pass

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
    
    def semantic_chunk_command(text: str, size: int, overlap: int) -> list[str]:
        return semantic_chunk(text, size, overlap)
    
    def embed_chunks_command():
        search = ChunkedSemanticSearch()
        documents = load_movies()
        embeddings = search.load_or_create_chunk_embeddings(documents)
        print(f"Generated {len(embeddings)} chunked embeddings")



    match args.command:
        case "search":
            search_command()
        case "search_chunked":
            search_chunked_command()
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "embed_chunks":
            embed_chunks_command()
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
        case "semantic_chunk":
            chunks = semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            n = 1
            for c in chunks:
                print(f"{n}. {c}")
                n += 1
        case "test":
            test_command()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()