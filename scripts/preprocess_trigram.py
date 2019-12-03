import sys
# usage: python preprocess_unigram [unigram_file_name]
# writes to ../trigrams


def read_and_parse_trigram(fileName):
    dest_file = open("../trigrams/" + fileName.split("/")[-1], "w+")
    trigram = []
    with open(fileName, "r") as ori_file:
        for line in ori_file.readlines():
            gram = line.strip()
            if len(gram) == 0:
                # in case of blank line
                continue
            trigram.append(gram)
            if len(trigram) >= 3:
                if len(trigram) > 3:
                    trigram = trigram[1:]
                if len(trigram) == 3:
                    dest_file.write(" ".join(trigram) + "\n")
    dest_file.close()


if __name__ == "__main__":
    file_name = sys.argv[1]
    read_and_parse_trigram(file_name)
