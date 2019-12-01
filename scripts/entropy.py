import utils
import math
import sys

def write_stats_to_file(stats, outfile="results/stats.csv"):
    with open(outfile, "w") as f:
        f.write("word\tfreq\tprob\tentropy\n")
        for word in sorted(stats.items(), key=lambda kv: kv[1][0], reverse=True):
            freq, prob, entropy = word[1]
            word = word[0]
            f.write(word + "\t" + str(freq) + "\t" +
                    str(prob) + "\t" + str(entropy) + "\n")

def calc_word_entropy(filename):
    data = utils.read_data(filename)
    wordfreq, total_counts = utils.data2wordfreq(data)
    entropy = 0
    stats = {}
    for word in wordfreq:
        freq = wordfreq[word]
        prob = freq / total_counts
        stats[word] = [freq, prob, -math.log(prob, 2)]
        entropy -= prob * math.log(prob, 2)
    write_stats_to_file(stats)
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
