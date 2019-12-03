import math
import os
import sys
import utils

def write_stats_to_file(stats, mega_stats, outfile="results/stats.csv"):
    wordfreq, total_counts, entropy = mega_stats
    with open(outfile, "w") as f:
        f.write("Overall entropy: {}\n".format(entropy))
        f.write("Size of tokens: {}\n".format(total_counts))
        f.write("Size of types: {}\n\n".format(len(wordfreq)))
        f.write("word\tfreq\tprob\tentropy\n")
        for word in sorted(stats.items(), key=lambda kv: kv[1][0], reverse=True):
            freq, prob, entropy = word[1]
            word = word[0]
            f.write(word + "\t" + str(freq) + "\t" +
                    str(prob) + "\t" + str(entropy) + "\n")

def calc_word_entropy(filename):
    # data = utils.read_data(filename)
    # wordfreq, total_counts = utils.data2wordfreq(data)
    ngrams = utils.read_ngrams(filename)
    wordfreq, total_counts = utils.ngram2wordfreq(ngrams)
    print("Size of types: {}".format(len(wordfreq)))
    print("Size of tokens: {}".format(total_counts))
    entropy = 0
    stats = {}
    for word in wordfreq:
        freq = wordfreq[word]
        prob = freq / total_counts
        stats[word] = [freq, prob, -math.log(prob, 2)]
        entropy -= prob * math.log(prob, 2)
    write_stats_to_file(stats, [wordfreq, total_counts, entropy], outfile="results/stats/" + filename.split("/")[-1])
    return entropy

def main(folder_file_name):
    if os.path.isfile(folder_file_name):
        print("=" * 5 + folder_file_name + "=" * 5)
        entropy = calc_word_entropy(folder_file_name)
        print("Entropy is {}".format(entropy))
    elif os.path.isdir(folder_file_name):
        for filename in os.listdir(folder_file_name):
            filename = os.path.join(folder_file_name, filename)
            print("=" * 5 + filename + "=" * 5)
            entropy = calc_word_entropy(filename)
            print("Entropy is {}".format(entropy))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python entropy.py [FOLDER|FILENAME]")
        exit(1)
    folder_file_name = sys.argv[1]
    main(folder_file_name)
