import numpy as np
import codecs
from collections import defaultdict
from numpy.random import permutation, shuffle, rand
from sklearn.model_selection import train_test_split


def prepare_train_data_for_task_1(include_dev=False):
    dataset = defaultdict(list)
    X, y = [], []

    with codecs.open(f'data/train.tsv') as training:
        for i, line in enumerate(training):
            sentence_label = line.strip().split('\t')
            X.append(sentence_label[0])
            y.append(sentence_label[1])
            dataset[sentence_label[1]].append(sentence_label[0])

    if include_dev:
        with codecs.open(f'data/dev.tsv') as training:
            for i, line in enumerate(training):
                sentence_label = line.strip().split('\t')
                X.append(sentence_label[0])
                y.append(sentence_label[1])
                dataset[sentence_label[1]].append(sentence_label[0])

    return train_test_split(X, y, test_size=0.0, shuffle=True)


def prepare_dev_data_for_task_1():
    dataset = defaultdict(list)
    X, y = [], []

    with codecs.open(f'data/dev.tsv') as training:
        for i, line in enumerate(training):
            sentence_label = line.strip().split('\t')
            X.append(sentence_label[0])
            y.append(sentence_label[1])
            dataset[sentence_label[1]].append(sentence_label[0])
    return X, y


def prepare_test_data_for_task_1():
    X = []
    with codecs.open(f'data/test.tsv') as training:
        for i, line in enumerate(training):
            sentence = line.strip()
            X.append(sentence)
    return X


if __name__ == '__main__':
    prepare_train_data_for_task_1()
