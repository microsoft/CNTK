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
import cntk as C
import numpy as np
from sklearn import datasets, utils
import scipy.sparse

# Define the task.
input_shape = (28, 28)  # MNIST digits are 28 x 28
num_classes = 10        # classify as one of 10 digits

# Fetch the MNIST data from mldata.
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

# Define the CNTK model function. The model function maps input data to
# predictions (here: 2-dimensional inputs --> 2 scores).
# This simple logistic-regression model just uses a linear transform,
# which corresponds to a Dense layer without activation function.
# A Dense layer implements the formula y = activation(x @ W + b), where W and b
# are learnable model parameters.
model = C.layers.Dense(num_classes, activation=None)

with C.layers.default_options(activation=C.ops.relu, pad=False):
    model = C.layers.Sequential([
        C.layers.Convolution2D((5,5), 32, reduction_rank=0, pad=True),
        C.layers.MaxPooling((3,3), (2,2)),
        C.layers.Convolution2D((3,3), 48),
        C.layers.MaxPooling((3,3), (2,2)),
        C.layers.Convolution2D((3,3), 64),
        C.layers.Dense(96),
        C.layers.Dropout(0.5),
        C.layers.Dense(num_classes, activation=None) # final softmax is added in criterion
    ])

# Define the CNTK criterion function. A criterion function maps
# (input vectors, labels) to a loss function and an optional additional
# metric. The loss function is used to train the model parameters.
# We use cross entropy as a loss function.
# We use CNTK @FunctionOf to declare a CNTK function with given input types.
# The cross-entropy formula requires the labels to be in one-hot format.
@C.FunctionOf(C.layers.Tensor[input_shape], C.layers.SparseTensor[num_classes])
def criterion(data, label_one_hot):
    z = model(data)  # apply model. Computes a non-normalized log probability for every output class.
    loss   = C.cross_entropy_with_softmax(z, label_one_hot) # this applies softmax to z under the hood
    metric = C.classification_error(z, label_one_hot)
    return loss, metric

# Learner object. The learner implements the update algorithm, in this case momentum SGD.
epoch_size = 60000
lr_per_sample    = [0.001]*10 + [0.0005]*10 + [0.0001]
lr_schedule      = C.learning_rate_schedule(lr_per_sample, C.learners.UnitType.sample, epoch_size)
mm_time_constant = [0]*5 + [1024]
mm_schedule      = C.learners.momentum_as_time_constant_schedule(mm_time_constant, epoch_size)

# Instantiate the trainer object to drive the model training
learner = C.learners.adam(model.parameters, lr_schedule, mm_schedule)

# Trainer configuration parameters.
progress_writer = C.logging.ProgressPrinter(50) # helper for logging progress; log every 50 minibatches

# Train!
progress = criterion.train((X_train, Y_train), minibatch_size=64, max_epochs=2, parameter_learners=[learner], progress_writers=[progress_writer])
final_loss, final_metric, final_samples = (progress.epoch_summaries[-1].loss, progress.epoch_summaries[-1].metric, progress.epoch_summaries[-1].samples)

# Test error rate on the test set.
test_metric = criterion.test((X_test, Y_test), progress_writers=[progress_writer]).metric

# Inspect predictions on one minibatch, for illustration.
# For evaluation, we map the output of the network between 0-1 and convert them into probabilities
# for the two classes. We use a softmax function to get the probabilities of each of the class.
@C.FunctionOf(C.layers.Tensor[input_shape])
def get_probability(data):
    return C.softmax(model(data))

X_check, Y_check = X_test[0:10000:400].copy(), Y_test[0:10000:400] # a small subsample of 25 examples
result = get_probability(X_check)

print("Label    :", [label.argmax() for label in Y_check])
print("Predicted:", [result[i,:].argmax() for i in range(len(result))])
