import numpy as np
import pandas as pd
import os
import sys
import utils
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


def build_model(X_train, y_train, vocab):
    # define the model
    model = Sequential()
    model.add(Embedding(vocab, 50, input_length=30, trainable=True))
    model.add(GRU(150, recurrent_dropout=0.1, dropout=0.1))
    model.add(Dense(vocab, activation='softmax'))
    # print(model.summary())
    # compile the model
    model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
    # fit the model
    model.fit(X_train, y_train, epochs=30, verbose=2, validation_split=0.1)
    return model

def predict_on_test(X_test, y_test, model, vocab):
    total = 0
    correct = 0
    for i, X in enumerate(X_test):
        X = pad_sequences([X])
        total += 1
        y = model.predict_classes(X, verbose=0)
        pred = to_categorical(y[0], num_classes=vocab)
        all_equal = True
        for j, yy in enumerate(pred):
            if yy != y_test[i][j]:
                all_equal = False
                break
        if all_equal:
            correct += 1
    print("Total: {}, correct: {}({}%)"
          .format(total, correct, correct / total * 100))

def neural_model_eval_pipeline(filename, vocab):
    X, y = utils.read_sequence(filename, vocab)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05,
                                                        random_state=42)
    model = build_model(X_train, y_train, vocab)
    predict_on_test(X_test, y_test, model, vocab)
    random.shuffle(y_test)
    predict_on_test(X_test, y_test, model, vocab)


def main(folder_file_name, mapping_name):
    if os.path.isfile(folder_file_name):
        mapping = utils.get_mapping(mapping_name)
        vocab = len(mapping) + 1
        print("=" * 5 + folder_file_name + "=" * 5)
        neural_model_eval_pipeline(folder_file_name, vocab)
    elif os.path.isdir(folder_file_name):
        for filename in os.listdir(folder_file_name):
            filename = os.path.join(folder_file_name, filename)
            mapping = utils.get_mapping(os.path.join(mapping_name,
                                                     filename.split("/")[-1]))
            vocab = len(mapping) + 1
            print("=" * 5 + filename + "=" * 5)
            neural_model_eval_pipeline(filename, vocab)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python neural_model.py [FOLDER|FILENAME] MAPPING_NAME")
        exit(1)
    folder_file_name = sys.argv[1]
    mapping_name = sys.argv[2]
    main(folder_file_name, mapping_name)



