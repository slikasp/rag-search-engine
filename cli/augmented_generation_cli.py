import argparse

from gemini import rag, summarize, cite, answer_question
from lib.load_data import load_movies
from lib.hybrid_search import HybridSearch

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="")
    summarize_parser.add_argument("query", type=str, help="")
    summarize_parser.add_argument("--limit", type=int, nargs='?', default=5, help="")

    citations_parser = subparsers.add_parser("citations", help="")
    citations_parser.add_argument("query", type=str, help="")
    citations_parser.add_argument("--limit", type=int, nargs='?', default=5, help="")

    question_parser = subparsers.add_parser("question", help="")
    question_parser.add_argument("query", type=str, help="")
    question_parser.add_argument("--limit", type=int, nargs='?', default=5, help="")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            documents = load_movies()
            search = HybridSearch(documents)
            movies = search.rrf_search(query, 60, 5)
            
            print("Search Results:")
            for m in movies:
                print(f"- {m[1]['title']}")
            print()
            print("RAG Response:")
            llm_response = rag(query, [m[1]['title'] for m in movies])
            print(llm_response)
        case "summarize":
            query = args.query
            limit = args.limit
            documents = load_movies()
            search = HybridSearch(documents)
            movies = search.rrf_search(query, 60, limit)

            print("Search Results:")
            for m in movies:
                print(f"- {m[1]['title']}")
            print()
            print("LLM Summary:")
            llm_response = summarize(query, movies)
            print(llm_response)
        case "citations":
            query = args.query
            limit = args.limit
            documents = load_movies()
            search = HybridSearch(documents)
            movies = search.rrf_search(query, 60, limit)

            print("Search Results:")
            for m in movies:
                print(f"- {m[1]['title']}")
            print()
            print("LLM Answer:")
            llm_response = cite(query, movies)
            print(llm_response)
        case "question":
            query = args.query
            limit = args.limit
            documents = load_movies()
            search = HybridSearch(documents)
            movies = search.rrf_search(query, 60, limit)

            print("Search Results:")
            for m in movies:
                print(f"- {m[1]['title']}")
            print()
            print("Answer:")
            llm_response = answer_question(query, movies)
            print(llm_response)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()