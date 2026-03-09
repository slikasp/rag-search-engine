import string

from load_data import load_stopwords

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stopwords = load_stopwords()

def preprocess(input: str) -> list[str]:
    input = input.lower()

    table = str.maketrans("", "", string.punctuation)
    input = input.translate(table)

    tokens = input.split()

    tokens = [token for token in tokens if token not in stopwords]

    stems = [stemmer.stem(token) for token in tokens]
    
    return stems