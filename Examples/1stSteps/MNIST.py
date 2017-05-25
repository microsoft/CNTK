# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# This example shows how to train a very basic CNTK model for logistic regression.
# The task is to classify a 2-dimensional vector as belong to one of two classes.
# The data is artificially created. Each class' data follows a normal distribution.

from __future__ import print_function
import os
import cntk
import numpy as np
from sklearn import datasets, utils
import scipy.sparse

# Define the task.
input_dim = (28, 28)  # MNIST digits are 28 x 28
num_classes = 10      # classify as one of 10 digits

def fetch_mnist():
    mnist = datasets.fetch_mldata("MNIST original")
    X, Y = mnist.data / 255., mnist.target
    X_train, X_test = X[:60000].reshape((-1,28,28)), X[60000:].reshape((-1,28,28))
    Y_train, Y_test = Y[:60000].astype(int), Y[60000:].astype(int)
    # Shuffle the training data.
    np.random.seed(0) # always use the same reordering, for reproducability
    X_train, Y_train = utils.shuffle(X_train, Y_train)
    # Our model expects float32 features, and cross-entropy expects one-hot encoded labels.
    Y_train, Y_test = (scipy.sparse.csr_matrix((np.ones(len(Y),np.float32), (range(len(Y)), Y)), shape=(len(Y), 10)) for Y in (Y_train, Y_test))
    X_train, X_test = (X.astype(np.float32) for X in (X_train, X_test))
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = fetch_mnist()

#import requests
#import gzip
#
#mnist_path = os.path.dirname(os.path.abspath(__file__)) + '/mnist'
#
## Download the files if not present yet.
#if not os.path.isfile(mnist_path)    or True:
#    data_arrays = {} # this code is strictly for the 4 MNIST files from Yann Lecun's web site
#    for name in ('train-images', 'train-labels', 't10k-images', 't10k-labels'):
#        is_image = name[-6:] == 'images'
#        open(mnist_path + ".tmp.gz", 'wb').write(requests.get('http://yann.lecun.com/exdb/mnist/' + name + '-idx' + "13"[is_image] + '-ubyte.gz').content)
#        data = gzip.GzipFile(mnist_path + ".tmp.gz", 'rb').read()[(8,16)[is_image]:]
#        os.remove(mnist_path + ".tmp.gz")
#        data_arrays[name.replace('-', '_')] = np.fromstring(data, dtype=np.uint8).reshape(((-1),(-1,28,28))[is_image]).astype((int, np.float32)[is_image])
#    np.savez_compressed(mnist_path, **data_arrays)

# Define the CNTK model function. The model function maps input data to
# predictions (here: 2-dimensional inputs --> 2 scores).
# This simple logistic-regression model just uses a linear transform,
# which corresponds to a Dense layer without activation function.
# A Dense layer implements the formula y = activation(x @ W + b), where W and b
# are learnable model parameters.
model = cntk.layers.Dense(num_classes, activation=None)

# Define the CNTK criterion function. A criterion function maps
# (input vectors, labels) to a loss function and an optional additional
# metric. The loss function is used to train the model parameters.
# We use cross entropy as a loss function.
# We use CNTK @FunctionOf to declare a CNTK function with given input types.
# The cross-entropy formula requires the labels to be in one-hot format.
@cntk.FunctionOf(cntk.layers.Tensor[input_dim], cntk.layers.SparseTensor[num_classes])
def criterion(data, label_one_hot):
    z = model(data)  # apply model. Computes a non-normalized log probability for every output class.
    loss   = cntk.cross_entropy_with_softmax(z, label_one_hot) # this applies softmax to z under the hood
    metric = cntk.classification_error(z, label_one_hot)
    return loss, metric

# Learner object. The learner implements the update algorithm, in this case plain SGD.
learning_rate = 0.1
learner = cntk.sgd(model.parameters, cntk.learning_rate_schedule(learning_rate, cntk.UnitType.minibatch))

# Trainer configuration parameters.
progress_writer = cntk.logging.ProgressPrinter(50) # helper for logging progress; log every 50 minibatches

# Train!
losses, metrics, num_samples = criterion.train((X_train, Y_train), parameter_learners=[learner], progress_writers=[progress_writer])

# Test error rate on the test set.
metric, num_samples = criterion.test((X_test, Y_test), progress_writers=[progress_writer])

# Inspect predictions on one minibatch, for illustration.
# For evaluation, we map the output of the network between 0-1 and convert them into probabilities
# for the two classes. We use a softmax function to get the probabilities of each of the class.
@cntk.FunctionOf(cntk.layers.Tensor[input_dim])
def get_probability(data):
    return cntk.softmax(model(data))

X_check, Y_check = X_test[0:10000:400], Y_test[0:10000:400] # a small subsample of 25 examples
result = get_probability(X_check)

print("Label    :", [label.argmax() for label in Y_check])
print("Predicted:", [result[i,:].argmax() for i in range(len(result))])




#def fetch_mnist():
#    mnist = datasets.fetch_mldata("MNIST original")
#    X, Y = mnist.data / 255., mnist.target
#    X_train, X_test = X[:60000].reshape((-1,28,28)), X[60000:].reshape((-1,28,28))
#    Y_train, Y_test = Y[:60000].astype(int), Y[60000:].astype(int)
#    # shuffle the training data
#    np.random.seed(0) # always use the same reordering, for reproducability
#    X_train, Y_train = utils.shuffle(X_train, Y_train)
#    # 
#    Y_train, Y_test = (scipy.sparse.csr_matrix((np.ones(len(Y),np.float32), (range(len(Y)), Y)), shape=(len(Y), 10)) for Y in (Y_train, Y_test))
#    X_train, X_test = (X.astype(np.float32) for X in (X_train, X_test))
#    return X_train, Y_train, X_test, Y_test
#
#X_train, Y_train, X_test, Y_test = fetch_mnist()
