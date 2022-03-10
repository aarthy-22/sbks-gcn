# Author : Samantha Mahendran for RelEx-GCN
# includes code from Yao, et al.(https://github.com/yao8839836/text_gcn)

from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from sklearn import metrics
from corpus_level.utils import *
from corpus_level.models import GCN
from eval.write_predictions import Predictions
import random
import os
import sys

# Set random seed
seed = random.randint(1, 200)
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# flags
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'gcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 15, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0,
                   'Weight for L2 loss on embedding matrix.')  # 5e-4
flags.DEFINE_integer('early_stopping', 15,
                     'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')


def sample_mask(idx, l):
    """
    Create mask.
    :rtype: object
    """
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def evaluate(features, support, labels, mask, placeholders, model, sess):
    """
    Define model evaluation function
    :param features:
    :param support:
    :param labels:
    :param mask:
    :param placeholders:
    :param model:
    :param sess:
    :return:
    """
    t_test = time.time()
    feed_dict_val = construct_feed_dict(
        features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], outs_val[3], (time.time() - t_test)


def load_graph(x, y, tx, ty, allx, ally, adj, train_ids):
    """
    load the buit graph
    :param x: train data
    :param y: train labels
    :param tx: test data
    :param ty: test labels
    :param allx: train + test + vocab
    :param ally: train + test + vocab labels
    :param adj: adjacency matrix
    :param train_ids: ids for the train data
    :return:
    """
    # merged list of train, test and vocab
    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))

    train_size = len(train_ids)
    val_size = train_size - x.shape[0]
    test_size = tx.shape[0]

    # range from 0 - length of the data
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    # true - when a label is present, false when it is not
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    # one-hot vector representation of the labels
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size


def train_test(features, y_train, y_val, y_test, train_mask, val_mask, test_mask, placeholders, model, support, sess):
    """

    :param features:
    :param y_train:
    :param y_val:
    :param y_test:
    :param train_mask:
    :param val_mask:
    :param test_mask:
    :param placeholders:
    :param model:
    :param support:
    :param sess:
    :return:
    """
    cost_val = []

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(
            features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy,
                         model.layers[0].embedding], feed_dict=feed_dict)

        # Validation
        cost, acc, pred, labels, duration = evaluate(
            features, support, y_val, val_mask, placeholders, model, sess)
        cost_val.append(cost)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(
                outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")

    test_cost, test_acc, pred, labels, test_duration = evaluate(
        features, support, y_test, test_mask, placeholders, model, sess)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

    test_pred = []
    test_labels = []

    for i in range(len(test_mask)):
        if test_mask[i]:
            test_pred.append(pred[i])
            test_labels.append(labels[i])

    return test_labels, test_pred


class TrainModel:

    def __init__(self, graph, write_Predictions=True, initial_predictions=None, final_predictions=None,
                 write_No_rel=False):
        self.graph = graph
        print("Training started !")
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_graph(
            self.graph.x, self.graph.y, self.graph.tx, self.graph.ty, self.graph.allx, self.graph.ally,
            self.graph.adj, self.graph.train_ids)

        features = sp.identity(features.shape[0])  # featureless

        # preprocessing to normalize feature matrix and convert to tuple representation
        features = preprocess_features(features)

        # parameters
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN

        # Define placeholders
        placeholders = {
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
            'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            # helper variable for sparse dropout
            'num_features_nonzero': tf.placeholder(tf.int32)
        }

        model = model_func(placeholders, input_dim=features[2][1], logging=True)

        # Initialize session
        session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        sess = tf.Session(config=session_conf)
        # Init variables
        sess.run(tf.global_variables_initializer())
        test_labels, test_pred = train_test(features, y_train, y_val, y_test, train_mask, val_mask, test_mask,
                                            placeholders, model, support,sess)
        print("Training completed !!!")
        if write_Predictions:
            # np.save('track', np.array(test_labels))
            np.save('pred', np.array(test_pred))
            Predictions(self.graph, test_pred, initial_predictions, final_predictions, write_No_rel)
        print("Predicitons written to external folder !!!")