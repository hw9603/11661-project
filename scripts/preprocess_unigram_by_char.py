import sys
# usage: python preprocess_unigram [file_name]
# writes to ../unigrams
# treat each character as a uni-gram


def read_and_parse_unigram_by_character(fileName):
    p = 1
    dest_file = open("../unigrams/" + fileName.split("/")[-1], "w+")
    with open(fileName, "r") as ori_file:
        for line in ori_file.readlines():
            line = line.strip()
            words = line.split()
            for word in words:
                for char in str(word):
                    dest_file.write(char + '\n')
    dest_file.close()


if __name__ == "__main__":
    file_name = sys.argv[1]
    read_and_parse_unigram_by_character(file_name)