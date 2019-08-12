import numpy as np
import codecs
from collections import defaultdict
from numpy.random import permutation, shuffle, rand
from sklearn.model_selection import train_test_split


def prepare_train_data_for_task_1(corpus, include_dev=False):
    dataset = defaultdict(list)
    X, y = [], []

    with codecs.open(f'{corpus}/data/train.tsv') as training:
        for i, line in enumerate(training):
            sentence_label = line.strip().split('\t')
            X.append(sentence_label[0])
            y.append(sentence_label[1])
            dataset[sentence_label[1]].append(sentence_label[0])

    if include_dev:
        with codecs.open(f'{corpus}/data/dev.tsv') as training:
            for i, line in enumerate(training):
                sentence_label = line.strip().split('\t')
                X.append(sentence_label[0])
                y.append(sentence_label[1])
                dataset[sentence_label[1]].append(sentence_label[0])

    return train_test_split(X, y, test_size=0.0, shuffle=True)


def prepare_dev_data_for_task_1(corpus):
    # sentences = list()
    # labels = list()
    #
    # labels_dist = set()

    # dataset = dict()
    # We will release training and testing data for the following Arabic dialects:
    # Egyptian, Gulf, Levantine, and North-African, and Modern Standard Arabic (MSA)

    # target_labels = ['RAB', 'MSA', 'BEI', 'DOH', 'CAI', 'TUN']

    dataset = defaultdict(list)
    X, y = [], []

    with codecs.open(f'{corpus}/data/dev.tsv') as training:
        for i, line in enumerate(training):
            sentence_label = line.strip().split('\t')
            X.append(sentence_label[0])
            y.append(sentence_label[1])
            dataset[sentence_label[1]].append(sentence_label[0])
    return X, y


def prepare_test_data_for_task_1(corpus):
    X = []
    with codecs.open(f'{corpus}/data/test.tsv') as training:
        for i, line in enumerate(training):
            sentence = line.strip()
            X.append(sentence)
    return X

# def divide_dataset(dataset, dev=0.2):
#
#     samples_train = dict()
#     samples_dev = dict()
#
#     for dialect, sentences in dataset.items():
#         samples = permutation(sentences)
#         dev_len = int(np.ceil(len(samples) * dev))
#         samples_dev[dialect] = sentences[:dev_len]
#         samples_train[dialect] = sentences[dev_len:]
        # dev_len = 0
        # if dev:
        #     devp = dev_perc / (100.0 - 60)
        #     dev_len = int(np.ceil((len(samples) - train_len) * devp))
        #     samples_dev[dialect] = sentences[train_len:train_len + dev_len]
        #     samples_test[dialect] = sentences[train_len + dev_len:]
        # else:
        #     samples_dev[dialect] = list()
        #     samples_test[dialect] = sentences[train_len:]
    # else:
    #     return samples_train, samples_dev


# train_set, dev_set, test_set = divide_dataset(dataset)