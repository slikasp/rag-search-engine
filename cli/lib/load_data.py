import json

def load_movies(path: str = "data/movies.json") -> dict:
    with open(path, 'r') as file:
        data = json.load(file)
    
    return data['movies']

def load_stopwords(path: str = "data/stopwords.txt") -> list[str]:
    with open(path, 'r') as f:
        words = f.read().splitlines()

    return words