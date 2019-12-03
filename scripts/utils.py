import string
from collections import defaultdict

def read_data(filename):
    data = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            # Remove leading and trailing whitespaces
            line = line.strip()
            # Remove punctuations
            line = line.translate(str.maketrans('', '', string.punctuation))
            data.append(line)
    return data

def data2wordfreq(data):
    wordfreq = defaultdict(int)
    total_counts = 0
    for line in data:
        # Assume the whitespace is the delimiter
        words = line.split(" ")
        for word in words:
            wordfreq[word] += 1
            total_counts += 1
    return wordfreq, total_counts

def read_ngrams(filename):
    ngrams = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            ngrams.append(line)
    return ngrams

def ngram2wordfreq(ngrams):
    wordfreq = defaultdict(int)
    total_counts = 0
    for ngram in ngrams:
        wordfreq[ngram] += 1
        total_counts += 1
    return wordfreq, total_counts
