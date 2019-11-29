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
