import argparse

from lib.multimodal_search import MultimodalSearch, verify_image_embedding, image_search_command

def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_embedding = subparsers.add_parser("verify_image_embedding", help="")
    verify_embedding.add_argument("image_path",type=str, default="data/paddington.jpeg", help="the path to an image file")

    image_search_embedding = subparsers.add_parser("image_search", help="")
    image_search_embedding.add_argument("image_path",type=str, default="data/paddington.jpeg", help="the path to an image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            movies = image_search_command(args.image_path)
            n = 1
            for m in movies:
                print(f"{n}. {m['title']} (similarity: {m['similarity']:.3f})")
                print(f"   {m['description']:.100}")
                n += 1
        case _:
            parser.print_help() 


if __name__ == "__main__":
    main()