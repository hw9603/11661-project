import utils
import math
import sys

def calc_word_entropy(filename):
    data = utils.read_data(filename)
    wordfreq, total_counts = utils.data2wordfreq(data)
    entropy = 0
    for word in wordfreq:
        freq = wordfreq[word]
        prob = freq / total_counts
        entropy -= prob * math.log(prob, 2)
    return entropy


def main(filename):
    entropy = calc_word_entropy(filename)
    print("Entropy is {}".format(entropy))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python entropy.py FILENAME")
        exit(1)
    filename = sys.argv[1]
    main(filename)
