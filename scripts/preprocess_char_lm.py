# preprocessing for character level language models
# writes to ../char_dict and ../char_data


def make_sequence(file_name, encode_dic, dest_file_name):
    sequence_len = 31
    dest_file = open(dest_file_name, "w+")
    with open(file_name, "r") as ori_file:
        temp = []
        for line in ori_file.readlines():
            line = line.strip()
            # read in characters, including spaces
            for char in line:
                temp.append(str(encode_dic[char]))
                if len(temp) == sequence_len:
                    dest_file.write(" ".join(temp) + "\n")
                    temp = temp[1:]
    dest_file.close()


def get_char_count_and_enocde(file_name, dict_file_name):
    dic = {}
    idx = 1
    with open(file_name, "r") as ori_file:
        for line in ori_file.readlines():
            line = line.strip()
            for char in line:
                if char not in dic:
                    dic[char] = idx
                    idx += 1
    # write dictionary to output file
    inv_dic = {v: k for k, v in dic.items()}
    with open(dict_file_name, "w+") as dic_file:
        for idx in range(1, len(dic) + 1):
            dic_file.write(inv_dic[idx] + "\n")
    return dic


def main(data_file_names, dest_file_names, dict_file_names):
    for i in range(len(data_file_names)):
        data_file_name, dest_file_name, dict_file_name\
            = data_file_names[i], dest_file_names[i], dict_file_names[i]
        char_dict = get_char_count_and_enocde(data_file_name, dict_file_name)
        make_sequence(data_file_name, char_dict, dest_file_name)


if __name__ == "__main__":
    data_file_names = ["../data/" + str(i) + ".txt" for i in range(1, 14)]
    dest_file_names = ["../char_data/" + str(i) + ".txt" for i in range(1, 14)]
    dict_file_names = ["../char_dict/" + str(i) + ".txt" for i in range(1, 14)]
    main(data_file_names, dest_file_names, dict_file_names)