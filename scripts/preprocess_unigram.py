import sys
# usage: python preprocess_unigram [file_name]
# writes to ../unigrams

def read_and_parse_unigram(fileName):
    dest_file = open("../unigrams/" + fileName.split("/")[-1], "w+")
    with open(fileName, "r") as ori_file:
        for line in ori_file.readlines():
            line = line.strip()
            unigrams = line.split()
            for unigram in unigrams:
                print(len(unigram), unigram, type(unigram))
                if unigram is None or len(unigram) == 0: continue
                dest_file.write('\n' + unigram)
    dest_file.close()


if __name__ == "__main__":
    file_name = sys.argv[1]
    read_and_parse_unigram(file_name)
