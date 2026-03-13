import string

from lib.load_data import load_stopwords

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stopwords = load_stopwords()

def test_stem(one, two):
    print(stemmer.stem(one))
    print(stemmer.stem(two))

def preprocess(input: str) -> list[str]:
    input = input.lower()

    table = str.maketrans("", "", string.punctuation)
    input = input.translate(table)

    tokens = input.split()

    tokens = [token for token in tokens if token not in stopwords]

    stems = [stemmer.stem(token) for token in tokens]
    
    return stems