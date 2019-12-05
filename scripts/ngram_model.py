import os
import random
import sys
import utils
from collections import defaultdict
from sklearn.model_selection import train_test_split

def build_ngram_model(X_train, y_train):
    model = defaultdict(lambda: defaultdict(lambda: 0))
    for i, X in enumerate(X_train):
        model[tuple(list(X))][y_train[i]] += 1
    for X in model:
        total_count = float(sum(model[X].values()))
        for y in model[X]:
            model[X][y] /= total_count
    return model

def predict_on_test(X_test, y_test, model, top_n=5):
    total = 0
    correct = 0
    empty = 0
    for i, X in enumerate(X_test):
        total += 1
        cand = model[tuple(list(X))]
        top_n_cand = sorted(cand.items(), key=lambda kv: kv[1], reverse=True)[0:top_n]
        if len(top_n_cand) == 0:
            empty += 1
        for c in top_n_cand:
            if c[0] == y_test[i]:
                correct += 1
                break
    print("Total: {}, correct: {}({}%), empty: {}({}%)"
          .format(total, correct, correct / total * 100,
                  empty, empty / total * 100))

def ngram_eval_pipeline(filename):
    ngrams = utils.read_ngrams(filename)
    X, y = utils.gen_ngram_label(ngrams)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.05,
                                                        random_state=42)
    model = build_ngram_model(X_train, y_train)
    # predict on structured test data
    predict_on_test(X_test, y_test, model)
    # Shuffle label for test
    random.shuffle(y_test)
    # predict on jumbled test data
    print("shuffle!")
    predict_on_test(X_test, y_test, model)

def main(folder_file_name):
    if os.path.isfile(folder_file_name):
        print("=" * 5 + folder_file_name + "=" * 5)
        ngram_eval_pipeline(folder_file_name)
    elif os.path.isdir(folder_file_name):
        for filename in os.listdir(folder_file_name):
            filename = os.path.join(folder_file_name, filename)
            print("=" * 5 + filename + "=" * 5)
            ngram_eval_pipeline(filename)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python ngram_model.py [FOLDER|FILENAME]")
        exit(1)
    folder_file_name = sys.argv[1]
    main(folder_file_name)
