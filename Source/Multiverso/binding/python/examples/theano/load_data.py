#!/usr/bin/env python
# coding:utf8

import cPickle
import os
import sys
CUR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
CIFAR10_DIR = os.path.abspath(os.path.join(CUR_DIR, os.path.pardir, 'data', 'cifar-10-batches-py'))


import numpy as np


def load_cifar10(data_dir=CIFAR10_DIR):
    '''
    we assume these files are in data_dir:
    batches.meta  data_batch_1 data_batch_2  data_batch_3  data_batch_4
    data_batch_5  readme.html test_batch

    You can download the data from
    https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

    The RGB values are scaled to [0., 1.].
    '''
    x_train_l = []
    t_train_l = []

    for i in xrange(1, 6):
        filename = os.path.join(data_dir, "data_batch_%d" % i)
        with open(filename, "rb") as f:
            data_obj = cPickle.load(f)
            x_train_l.append(data_obj["data"])
            t_train_l.extend(data_obj["labels"])
    x_train = np.concatenate(x_train_l, axis=0) / 255.

    t_train = np.zeros((x_train.shape[0], 10))
    for i, cls in enumerate(t_train_l):
        t_train[i, cls] = 1

    with open(os.path.join(data_dir, "test_batch")) as f:
        data_obj = cPickle.load(f)
        x_test = data_obj["data"] / 255.
        t_test_l = data_obj["labels"]

        t_test = np.zeros((x_test.shape[0], 10))
        for i, cls in enumerate(t_test_l):
            t_test[i, cls] = 1
    return x_train, t_train, x_test, t_test
