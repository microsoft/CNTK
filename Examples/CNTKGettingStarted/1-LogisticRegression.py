# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# This example shows how to train a very basic CNTK model for logistic regression.
# The task is to classify a 2-dimensional vector as belong to one of two classes.
# The data is artificially created. Each class' data follows a normal distribution.

# Import the relevant components
from __future__ import print_function
import os
import argparse
import math
import cntk
from cntk.layers import *
from cntk.layers.typing import *

# Define the task
input_dim = 2    # classify 2-dimensional data
num_classes = 2  # into one of two classes

num_samples_to_train = 20000
num_samples_to_test  = 20000

# Generate our synthetic data
#  X[sample_size,input_dim] - our input data
#  Y[sample_size]           - labels (0 or 1), in one-hot representation
sample_size = 32
np.random.seed(0)
def generate_synthetic_data(N):
    # labels
    Y = np.random.randint(size=N, low=0, high=num_classes)
    # data
    X = (np.random.randn(N, input_dim)+3) * (Y[:,None]+1)
    # our model expects float32 features, and cross-entropy expects one-hot encoded labels
    from scipy import sparse
    Y = sparse.csr_matrix((np.ones(N,np.float32), (range(N), Y)), shape=(N, num_classes))
    X = X.astype(np.float32)
    return X, Y
X_train, Y_train = generate_synthetic_data(num_samples_to_train)
X_test,  Y_test  = generate_synthetic_data(num_samples_to_test)

# Define the CNTK model. A model is defined as a function that maps
# input data to some form of predictions, in our case 2-dimensional
# input vectors to a 2-dimensional vector of scores.
# This simple logistic-regression model just uses a linear transform,
# which corresponds to a Dense layer without activation function.
# A Dense layer implements the formula y = x @ W + b, where W and b
# are learnable model parameters.
model = cntk.layers.Dense(num_classes, activation=None)

# Define the CNTK criterion function. A criterion function maps
# (input vectors, labels) to a loss function and an optional additional
# metric. The loss function is used to train the model parameters.
# We use cross entropy as a loss function.
# We use CNTK @Signature to declare the input types at this point.
# The cross-entropy formula requires the labels to be in one-hot format.
@cntk.Function
@Signature(cntk.layers.Tensor[input_dim], cntk.layers.SparseTensor[num_classes])
def criterion(data, label_one_hot):
    z = model(data)  # apply model. Computes a non-normalized log probability for every output class.
    loss   = cntk.cross_entropy_with_softmax(z, label_one_hot)
    metric = cntk.classification_error(z, label_one_hot)
    return loss, metric

# Instantiate the trainer object to drive the model training
learning_rate = 0.5
learner = cntk.sgd(model.parameters, cntk.learning_rate_schedule(learning_rate, cntk.UnitType.minibatch))
trainer = cntk.Trainer(None, criterion, [learner], progress_writers=[cntk.logging.ProgressPrinter()])

# Initialize the parameters for the trainer
minibatch_size = 25

minibatch_source = cntk.io.MinibatchSourceFromData(data=X_train, labels=Y_train)
model_inputs_to_streams = {criterion.arguments[0]: minibatch_source.streams.data, criterion.arguments[1]: minibatch_source.streams.labels}

cntk.training_session(trainer, minibatch_source, minibatch_size,
                   model_inputs_to_streams,
                   progress_frequency=minibatch_size * 50, max_samples=len(X_train),
                   #test_config=cntk.TestConfig(source=cntk.io.MinibatchSourceFromData(data=X_test, labels=Y_test), mb_size=1000)
                   # currently crashing, to be fixed once I get a response to my bug report
                   ).train()

# Checking prediction on one minibatch
# For evaluation, we map the output of the network between 0-1 and convert them into probabilities
# for the two classes. We use a softmax function to get the probabilities of each of the class.
@cntk.Function
@Signature(cntk.layers.Tensor[input_dim])
def get_probability(data):
    return cntk.softmax(model(data))

X_check,  Y_check = generate_synthetic_data(25) # a small batch of 25 examples
result = get_probability(X_check)

print("Label    :", [label.argmax() for label in Y_check])
print("Predicted:", [result[i,:].argmax() for i in range(len(result))])
