# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# This example shows how to train a very basic CNTK model for logistic regression.
# The task is to classify a 2-dimensional vector as belonging to one of two classes.
# The data is artificially created.
# This example is identical to LogisticRegression_FunctionalAPI.py, except that
# it uses lower-level APIs.

from __future__ import print_function
import cntk
import numpy as np
import scipy.sparse

# Define the task.
input_dim = 2    # classify 2-dimensional data
num_classes = 2  # into one of two classes

# This example uses synthetic data from a normal distribution, which we generate in the following.
#  X[corpus_size,input_dim] - our input data
#  Y[corpus_size]           - labels (0 or 1), in one-hot representation
np.random.seed(0)
def generate_synthetic_data(N):
    Y = np.random.randint(size=N, low=0, high=num_classes)  # labels
    X = (np.random.randn(N, input_dim)+3) * (Y[:,None]+1)   # data
    # Our model expects float32 features, and cross-entropy expects one-hot encoded labels.
    Y = scipy.sparse.csr_matrix((np.ones(N,np.float32), (range(N), Y)), shape=(N, num_classes))
    X = X.astype(np.float32)
    return X, Y
X_train, Y_train = generate_synthetic_data(20000)
X_test,  Y_test  = generate_synthetic_data(1024)

# Define the CNTK model function. The model function maps input data to
# predictions (here: 2-dimensional inputs --> 2 scores).
# This simple logistic-regression model just uses a linear transform.
data = cntk.input_variable(input_dim)
W = cntk.Parameter((input_dim, num_classes), init=cntk.glorot_uniform(), name='W')
b = cntk.Parameter((num_classes,), init=0, name='b')
model = cntk.times(data, W) + b

# Define the CNTK criterion function. A criterion function maps
# (input vectors, labels) to a loss function and an optional additional
# metric. The loss function is used to train the model parameters.
# We use cross entropy as a loss function.
label_one_hot = cntk.input_variable(num_classes, is_sparse=True)
loss   = cntk.cross_entropy_with_softmax(model, label_one_hot) # this applies softmax to model's output under the hood
metric = cntk.classification_error(model, label_one_hot)
criterion = cntk.combine([loss, metric]) # criterion is a tuple-valued function (loss, metric)

# Learner object. The learner implements the update algorithm, in this case plain SGD.
learning_rate = 0.1
learner = cntk.sgd(model.parameters, cntk.learning_parameter_schedule(learning_rate))

# Trainer.
minibatch_size = 32
progress_writer = cntk.logging.ProgressPrinter(50) # helper for logging progress; log every 50 minibatches
trainer = cntk.Trainer(None, criterion, [learner], [progress_writer])

# Train!
for i in range(0, len(X_train), minibatch_size): # loop over minibatches
    x = X_train[i:i+minibatch_size] # get one minibatch worth of data
    y = Y_train[i:i+minibatch_size]
    trainer.train_minibatch({data: x, label_one_hot: y})  # update model from one minibatch
trainer.summarize_training_progress()

# Test error rate on the test set.
evaluator = cntk.Evaluator(metric, [progress_writer])
for i in range(0, len(X_test), minibatch_size): # loop over minibatches
    x = X_test[i:i+minibatch_size] # get one minibatch worth of data
    y = Y_test[i:i+minibatch_size]
    evaluator.test_minibatch({data: x, label_one_hot: y})  # test one minibatch
evaluator.summarize_test_progress()

# Inspect predictions on one minibatch, for illustration.
# For evaluation, we map the output of the network between 0-1 and convert them into probabilities
# for the two classes. We use a softmax function to get the probabilities of each of the class.
get_probability = cntk.softmax(model)

X_check, Y_check = generate_synthetic_data(25) # a small batch of 25 examples
result = get_probability.eval(X_check)

print("Label    :", [label.todense().argmax() for label in Y_check])
print("Predicted:", [result[i,:].argmax() for i in range(len(result))])
