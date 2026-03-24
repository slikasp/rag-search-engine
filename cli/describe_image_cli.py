import argparse
import mimetypes
from gemini import image
from lib.load_data import load_movies
from lib.hybrid_search import HybridSearch

def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    parser.add_argument("--image",type=str, default="data/paddington.jpeg", help="the path to an image file")
    parser.add_argument("--query",type=str, default="", help="a text query to rewrite based on the image")

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, mode='rb') as f:
        img = f.read()

    llm_response = image(args.query, img, mime)
    
    print(f"Rewritten query: {llm_response}")


if __name__ == "__main__":
    main()