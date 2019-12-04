import csv
import pandas
import statsmodels.api as sm
import matplotlib.pyplot as plt
import sys
import string
import numpy as np
import math
from matplotlib.backends.backend_pdf import PdfPages


# returns {word: freq}, and total word number
def get_word_freq(data_file):
    ori_file = open(data_file, 'r')
    dic = {}
    tot = 0
    for line in ori_file.readlines():
        word = line.strip()
        tot += 1
        if word in dic:
            dic[word] += 1
        else:
            dic[word] = 1
    ori_file.close()
    # print(type(dic))
    return dic, tot, len(dic.keys())


# returns {freq:word}
def get_rank_freq(data_file):
    ori_file = open(data_file, 'r')
    dic = {}
    tot = 0
    for line in ori_file.readlines():
        word = line.strip()
        tot += 1
        dic[word] = dic.get(word, 0) + 1
    ori_file.close()
    # print(tot)

    sorted_all = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    freqs = []
    ranks = []
    # print(len(sorted_all))

    for (idx, word) in enumerate(sorted_all):
        freqs.append(sorted_all[idx][1])
        ranks.append(idx + 1)
    return ranks, freqs


def plot(ranks, freqs, filename):
    filename_stripped = filename.split('/')[-1]
    plt.title(" Word Frequencies in " + filename_stripped)
    plt.ylabel("log frequency")
    plt.xlabel("log rank")

    # actual data
    log_ranks = [math.log(x, 10) for x in ranks]
    log_freqs = [math.log(x, 10) for x in freqs]
    plt.plot(
        log_ranks,
        log_freqs,
    )

    # OLS fitted data
    log_ranks_ols = sm.add_constant(log_ranks)
    model = sm.OLS(log_freqs, log_ranks_ols)
    results = model.fit()
    k, b = results.params[1], results.params[0]
    std_err = results.bse[0]
    log_freqs_ols = [k * i + b for i in log_ranks]
    plt.plot(log_ranks, log_freqs_ols)

    plt.legend(['Word frequency',
                'OLS, slope={:0.3f}, offset={:0.3f}, std_err={:0.3f}'.format(k, b, std_err)
                ],
               loc='upper left')
    plt.savefig('../results/zipfs_law/' + filename.split('/')[-1].split('.')[-2] + '.pdf')
    plt.clf()
    return std_err, filename_stripped


def main(data_file_names, stats_file_name):
    with open(stats_file_name, "w+") as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'std_err'])
        for filename in data_file_names:
            ranks, freqs = get_rank_freq(filename)
            std_err, filename_stripped = plot(ranks, freqs, filename)
            writer.writerow([filename_stripped, str(std_err)])


if __name__ == '__main__':
    data_file_names = ["../unigrams/" + str(i) + ".txt" for i in range(1, 14)]
    stats_file_name = "../results/zipfs_law/stats.csv"
    main(data_file_names, stats_file_name)
