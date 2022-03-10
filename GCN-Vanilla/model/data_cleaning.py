# Author : Samantha Mahendran for RelEx-GCN
from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from utils import clean_str, loadWord2Vec
import sys


def remove_words(doc_content_list, stop_words, word_freq, unique_entites):
    """
    Remove stop words and very low frequent words
    :type unique_entites: entity list
    :param doc_content_list: words in data
    :param stop_words: nltk stop words
    :param word_freq: word count list of the data
    :return: clean data
    """
    clean_docs = []
    for doc_content in doc_content_list.split('\n'):
        temp = clean_str(doc_content)
        words = temp.split()
        doc_words = []
        for word in words:
            # if word not in stop_words:
            if word not in stop_words:
                # if word in unique_entites:
                #     doc_words.append(word)
                # elif word_freq[word] >= 5:
                doc_words.append(word)
        doc_str = ' '.join(doc_words).strip()
        clean_docs.append(doc_str)

    # show statistics
    min_len = 10000
    aver_len = 0
    max_len = 0

    for line in clean_docs:
        line = line.strip()
        temp = line.split()
        aver_len = aver_len + len(temp)
        if len(temp) < min_len:
            min_len = len(temp)
        if len(temp) > max_len:
            max_len = len(temp)
    aver_len = 1.0 * aver_len / len(clean_docs)
    print('min_len : ' + str(min_len))
    print('max_len : ' + str(max_len))
    print('average_len : ' + str(aver_len))
    return clean_docs


# to remove rare words
def count_words(doc_content_list):
    word_freq = {}
    # count freq to find the low freq words
    for doc_content in doc_content_list.split('\n'):
        # print(doc_content)
        temp = clean_str(doc_content)
        words = temp.split()
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    return word_freq


class DataCleaning:

    def __init__(self, model):
        """
        :param model: model class object

        """
        self.data_model = model
        # load from stopwords
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        # count words in the data
        word_freq = count_words(self.data_model.data_str)
        # remove low freq words
        self.clean_docs = remove_words(self.data_model.data_str, stop_words, word_freq, self.data_model.unique_entities)
        print("Data cleaning completed !!!")