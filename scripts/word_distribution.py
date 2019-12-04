import csv
import pandas

# import csv
# with open('some.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerows(list)


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
def get_freq_word(word_freq_dic):
    dic = {}
    for word in word_freq_dic:
        cnt = word_freq_dic[word]
        if cnt in dic:
            dic[cnt].append(word)
        else:
            dic[cnt] = [word]
    return dic


def main(data_file_names, stats_file_name):
    k = 30
    beta = 0.5
    with open(stats_file_name, "w+") as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'total_words','unique_word_percentage',
                         'actual_distinct', 'predicted_distinct', 'differnce'])
        for data_file_name in data_file_names:
            word_freq, tot, tot_distinct = get_word_freq(data_file_name)
            # in case no word occurs only once

            # calculate unique word percentage
            uniq_words = get_freq_word(word_freq).get(1, [])
            uniq_prec = (len(uniq_words) * 1.0) / tot

            # calculate Heaps' Law predicted distinct count
            pred = k * (tot ** beta)
            # calculate prediction difference
            diff = ((pred - tot_distinct) * 1.0) / tot_distinct

            writer.writerow([data_file_name.split('/')[-1], str(int(tot)), str(uniq_prec),
                             str(tot_distinct), str(pred), str(diff)])



if __name__ == "__main__":
    # default data file names
    data_file_names = ["../unigrams/" + str(i) + ".txt" for i in range(1, 14)]
    stats_file_name = "../results/word_distribution/stats.csv"
    main(data_file_names, stats_file_name)
