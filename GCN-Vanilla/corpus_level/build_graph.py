# Author : Samantha Mahendran for RelEx-GCN
# includes code from Yao, et al.(https://github.com/yao8839836/text_gcn)

import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from utils import loadWord2Vec, clean_str
# from gensim.models import KeyedVectors
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine


def read_word2vec(embedding_path, binary=True):
    """
    Read external word embeddings
    :param embedding_path: path to the embeddings
    :return: embedding dimension. word to vec map
    """
    # if binary:
    #     word_vector_map = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
    #     word_embeddings_dim = len(word_vector_map[0])
    #
    # else:
    #     _, embd, word_vector_map = loadWord2Vec(embedding_path)
    #     word_embeddings_dim = len(embd[0])
    _, embd, word_vector_map = loadWord2Vec(embedding_path)
    word_embeddings_dim = len(embd[0])

    return word_embeddings_dim, word_vector_map


def split_train_test(records):
    """
    split train and test into separate lists. returns the label list.
    :param records: input data
    :return:
    """
    doc_name_list = []
    doc_train_list = []
    doc_test_list = []

    for line in records.split('\n'):
        doc_name_list.append(line.strip())
        temp = line.split("\t")
        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())

    label_set = set()
    for doc_meta in doc_train_list:
        temp = doc_meta.split('\t')
        label_set.add(temp[2])

    # list of labels for training
    label_list = list(label_set)

    return doc_name_list, doc_train_list, doc_test_list, label_list


def shuffle_ids(doc_list, doc_name_list):
    """
    shuffle ids of the data
    :param doc_list: merged data list
    :param doc_name_list:train or test data list
    :return:
    """
    ids = []
    for name in doc_list:
        id = doc_name_list.index(name)
        ids.append(id)
    random.shuffle(ids)
    return ids


def shuffle_data(ids, doc_name_list, doc_content_list):
    """
    re-arrange data according to the shuffled ids
    :param ids: shuffled ids
    :param doc_name_list: meta data list (id |train or test| label |track info)
    :param doc_content_list: sentences-only list
    :return:
    """
    shuffle_doc_name_list = []  # shuffled meta data
    shuffle_doc_words_list = []  # shuffled data

    for id in ids:
        shuffle_doc_name_list.append(doc_name_list[int(id)])
        shuffle_doc_words_list.append(doc_content_list[int(id)])

    return shuffle_doc_name_list, shuffle_doc_words_list


def build_vocab(shuffle_doc_words_list):
    """
    build the vocabulary (unique words) from the data
    :param shuffle_doc_words_list: words in the data
    :return:
    """
    # build vocab
    word_set = set()
    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        for word in words:
            word_set.add(word)
    vocab = list(word_set)
    vocab_size = len(vocab)

    # word and set of locations of the word in the data
    word_doc_list = {}

    # track the location of the words - position embeddings
    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        appeared = set()  # unique words
        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [i]
            appeared.add(word)

    # frequency of the words in the data
    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i

    return vocab, word_id_map, word_doc_freq


def generate_sent_vectors(data_size, word_embeddings_dim, shuffle_doc_words_list, word_vector_map):
    """
    generate sentence vectors and create the sparse matrix for sentence and unique words
    dimension (data size * word embeddings dimension)
    :param data_size: size of the data
    :param word_embeddings_dim: dimension of the word embeddings
    :param shuffle_doc_words_list: shuffled data
    :param word_vector_map: embeddings vector map
    :return: sparse matrix
    """
    row = []
    col = []
    data = []

    for i in range(data_size):
        sent_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        sent_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                # print(np.array(word_vector))
                sent_vec = sent_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row.append(i)
            col.append(j)
            # np.random.uniform(-0.25, 0.25)
            data.append(sent_vec[j] / sent_len)  # doc_vec[j]/ doc_len

    x = sp.csr_matrix((data, (row, col)), shape=(data_size, word_embeddings_dim))
    return x


def generate_label_vector(data_size, shuffle_doc_name_list, label_list):
    """
    generate one hot vectors for the labels
    :param data_size:
    :param shuffle_doc_name_list: shuffled data
    :param label_list: list of given labels of the training data
    :return: label vector as an array
    """
    y = []
    for i in range(data_size):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        # print(label_list)
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        y.append(one_hot)
    y = np.array(y)
    return y


def train_validation_split(train_ids, test_ids, vocab, shuffle_doc_name_list, shuffle_doc_words_list,
                           word_embeddings_dim, word_vector_map, label_list):
    """
    prepare the train, test and validation sets for the training
    :param train_ids: train ids
    :param test_ids: test
    :param vocab: vocabulary
    :param shuffle_doc_name_list:
    :param shuffle_doc_words_list:
    :param word_embeddings_dim:
    :param label_list:
    :return:
    """
    # select 90% training set
    train_size = len(train_ids)
    val_size = int(0.1 * train_size)
    real_train_size = train_size - val_size  # - int(0.5 * train_size)

    # tx: feature vectors of test docs, no initial features
    test_size = len(test_ids)

    # different training rates
    real_train_doc_names = shuffle_doc_name_list[:real_train_size]  # training set except for the validation set

    # generate sentence vectors for the train set
    x = generate_sent_vectors(real_train_size, word_embeddings_dim, shuffle_doc_words_list, word_vector_map)

    # generate label vectors for the train set
    y = generate_label_vector(real_train_size, shuffle_doc_name_list, label_list)

    # generate sentence vectors for the test set
    tx = generate_sent_vectors(test_size, word_embeddings_dim, shuffle_doc_words_list, word_vector_map)

    # generate label vectors for the test set
    ty = generate_label_vector(test_size, shuffle_doc_name_list, label_list)

    vocab_size = len(vocab)
    word_vectors = np.random.uniform(-0.01, 0.01, (vocab_size, word_embeddings_dim))

    # allx: the the feature vectors of both labeled and unlabeled training instances
    # (a superset of x)
    # unlabeled training instances -> words
    for i in range(len(vocab)):
        word = vocab[i]
        if word in word_vector_map:
            vector = word_vector_map[word]
            word_vectors[i] = vector

    # merge train, test and unique word vectors into the adjacency matrix
    row_allx = []
    col_allx = []
    data_allx = []

    for i in range(train_size):
        sent_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        sent_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                sent_vec = sent_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_allx.append(int(i))
            col_allx.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_allx.append(sent_vec[j] / sent_len)  # doc_vec[j]/doc_len

    for i in range(vocab_size):
        for j in range(word_embeddings_dim):
            row_allx.append(int(i + train_size))
            col_allx.append(j)
            data_allx.append(word_vectors.item((i, j)))

    row_allx = np.array(row_allx)
    col_allx = np.array(col_allx)
    data_allx = np.array(data_allx)

    allx = sp.csr_matrix(
        (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

    # merge one-hot vectors of train, test and words for labels
    ally = []
    for i in range(train_size):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        ally.append(one_hot)

    for i in range(vocab_size):
        one_hot = [0 for l in range(len(label_list))]
        ally.append(one_hot)

    ally = np.array(ally)

    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape, "=========================")
    return real_train_doc_names, x, y, tx, ty, allx, ally


def build_SentWord_graph(train_ids, test_ids, shuffle_doc_words_list, word_id_map, vocab, word_doc_freq,
                         window_size=20):
    """
    build sentence-word heterogenous graph with edges between sentence-words and between word-word.
    Sentence-word edge is calculated using TF-IDF and word-word edges are calculated using PMI.
    :param train_ids: Ids of train data
    :param test_ids: Ids of test data
    :param shuffle_doc_words_list: shuffled data
    :param word_id_map: word to ID reference
    :param vocab: unique words set
    :param word_doc_freq:
    :param window_size:
    :return:
    """
    train_size = len(train_ids)
    test_size = len(test_ids)
    vocab_size = len(vocab)

    # select words within the window size
    windows = []
    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            # print(length, length - window_size + 1)
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)
                # print(window)

    # calculate the frequency of the words inside a window size
    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])

    # calculate the times a word pair occured together
    word_pair_count = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word_id_map[word_i]
                word_j = window[j]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1

    # pmi calculation between words
    row = []
    col = []
    weight = []

    # pmi as weights
    num_window = len(windows)

    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(train_size + i)
        col.append(train_size + j)
        weight.append(pmi)

    # doc word frequency
    doc_word_freq = {}
    # calculate sent-word pair frequency
    for doc_id in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[doc_id]
        words = doc_words.split()
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1

    # TF-IDF calculation for sent - word pair
    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            if i < train_size:
                row.append(i)
            else:
                row.append(i + vocab_size)
            col.append(train_size + j)
            idf = log(1.0 * len(shuffle_doc_words_list) /
                      word_doc_freq[vocab[j]])
            weight.append(freq * idf)
            doc_word_set.add(word)

    node_size = train_size + vocab_size + test_size

    # adjacency matrix to build the graph
    adj = sp.csr_matrix(
        (weight, (row, col)), shape=(node_size, node_size))
    print(" Shape of the adjacency matrix :", adj.shape)
    return adj


class BuildGraph:

    def __init__(self, model, cleanData, window_size, embedding_binary, word_embeddings_path):
        """
        Build graph object
        :type model: object
        :type cleanData: object
        :param window_size: number of words considered as associated
        :param word_embeddings_dim: words in data

        """
        self.data_model = model
        self.cleanData = cleanData

        # read word2vec embeddings
        word_embeddings_dim, word_vector_map = read_word2vec(word_embeddings_path, embedding_binary)

        # split train - test data and shuffle the data
        doc_name_list, doc_train_list, doc_test_list, self.label_list = split_train_test(self.data_model.meta_data_str)
        self.train_ids = shuffle_ids(doc_train_list, doc_name_list)
        self.test_ids = shuffle_ids(doc_test_list, doc_name_list)
        self.ids = self.train_ids + self.test_ids
        shuffle_doc_name_list, shuffle_doc_words_list = shuffle_data(self.ids, doc_name_list, self.cleanData.clean_docs)
        print("Data preparation completed")

        # build word vectors
        vocab, word_id_map, word_doc_freq = build_vocab(shuffle_doc_words_list)

        real_train_doc_names, self.x, self.y, self.tx, self.ty, self.allx, self.ally = train_validation_split(
            self.train_ids, self.test_ids, vocab, shuffle_doc_name_list, shuffle_doc_words_list, word_embeddings_dim,
            word_vector_map, self.label_list)
        # build the sentence-words graph
        self.adj = build_SentWord_graph(self.train_ids, self.test_ids, shuffle_doc_words_list, word_id_map, vocab,
                                        word_doc_freq, int(window_size))
        print("Graph building completed !!!")
        self.shuffle_text = shuffle_doc_name_list
