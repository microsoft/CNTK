# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# This example demonstrates how to train a model, showcasing a range of training options: 
#  - checkpointing
#  - testing after each minibatch
#  - cross-validation based learning-rate control and early stopping in user code
#  - data-parallel distributed training using MPI
# This is shown along the task of recognizing handwritten digits on the MNIST corpus.

from __future__ import print_function
import os
import cntk as C
import numpy as np
import scipy.sparse

# Define the task.
input_shape = (28, 28)  # MNIST digits are 28 x 28
num_classes = 10        # classify as one of 10 digits
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models/mnist.cmf")

# Fetch the MNIST data.
# This requires scikit-learn, which is included in our recommended Python
# distribution (Anaconda). If you do not have it, please install it using
# pip (pip install -U scikit-learn) or conda (conda install scikit-learn).
try:
    from sklearn import datasets, utils
    mnist = datasets.fetch_mldata("MNIST original")
    X, Y = mnist.data / 255.0, mnist.target
    X_train, X_test = X[:60000].reshape((-1,28,28)), X[60000:].reshape((-1,28,28))
    Y_train, Y_test = Y[:60000].astype(int), Y[60000:].astype(int)
except: # workaround if scikit-learn is not present
    import requests, io, gzip
    X_train, X_test = (np.fromstring(gzip.GzipFile(fileobj=io.BytesIO(requests.get('http://yann.lecun.com/exdb/mnist/' + name + '-images-idx3-ubyte.gz').content)).read()[16:], dtype=np.uint8).reshape((-1,28,28)).astype(np.float32) / 255.0 for name in ('train', 't10k'))
    Y_train, Y_test = (np.fromstring(gzip.GzipFile(fileobj=io.BytesIO(requests.get('http://yann.lecun.com/exdb/mnist/' + name + '-labels-idx1-ubyte.gz').content)).read()[8:], dtype=np.uint8).astype(int) for name in ('train', 't10k'))

# Shuffle the training data.
np.random.seed(0) # always use the same reordering, for reproducability
idx = np.random.permutation(len(X_train))
X_train, Y_train = X_train[idx], Y_train[idx]

# Further split off a cross-validation set
X_train, X_cv = X_train[:54000], X_train[54000:]
Y_train, Y_cv = Y_train[:54000], Y_train[54000:]

# Our model expects float32 features, and cross-entropy expects one-hot encoded labels.
Y_train, Y_cv, Y_test = (scipy.sparse.csr_matrix((np.ones(len(Y),np.float32), (range(len(Y)), Y)), shape=(len(Y), 10)) for Y in (Y_train, Y_cv, Y_test))
X_train, X_cv, X_test = (X.astype(np.float32) for X in (X_train, X_cv, X_test))

# Define the CNTK model function. The model function maps input data to
# predictions (here: (28,28)-dimensional inputs --> 10 scores).
# This specific model uses convolution, max pooling, and dropout in a
# typical configuration.
with C.layers.default_options(activation=C.ops.relu, pad=False):
    model = C.layers.Sequential([
        C.layers.Convolution2D((5,5), num_filters=32, reduction_rank=0, pad=True), # reduction_rank=0 for B&W images
        C.layers.MaxPooling((2,2), strides=(2,2)),
        C.layers.Convolution2D((3,3), num_filters=48),
        C.layers.MaxPooling((2,2), strides=(2,2)),
        C.layers.Convolution2D((3,3), num_filters=64),
        C.layers.Dense(96),
        C.layers.Dropout(dropout_rate=0.5),
        C.layers.Dense(num_classes, activation=None) # no activation in final layer (softmax is done in criterion)
    ])

# Define the CNTK criterion function. A criterion function maps
# (input vectors, labels) to a loss function and an optional additional
# metric. The loss function is used to train the model parameters.
# We use cross entropy as a loss function.
# We use CNTK @Function.with_signature to declare a CNTK function with given input types.
# The cross-entropy formula requires the labels to be in one-hot format.
@C.Function.with_signature(C.layers.Tensor[input_shape], C.layers.SparseTensor[num_classes])
def criterion(data, label_one_hot):
    z = model(data)  # apply model. Computes a non-normalized log probability for every output class.
    loss   = C.cross_entropy_with_softmax(z, label_one_hot) # this applies softmax to z under the hood
    metric = C.classification_error(z, label_one_hot)
    return loss, metric

# Learner object. The learner implements the update algorithm, in this case momentum SGD.
# Because this script supports data-parallel training, the learning rate is specified
# "per sample", the value is already pre-divided by the minibatch size.
# This allows data-parallel training to slice the data into subsets and also to increase
# the minibatch size where possible, while maintaining the same contribution per sample gradient.
epoch_size = len(X_train)
lr_per_sample    = 0.001
lr_schedule      = C.learning_parameter_schedule_per_sample(lr_per_sample)
mm_per_sample    = [0]*5 + [0.9990239141819757] # 5 epochs without momentum, then switch it on
mm_schedule      = C.learners.momentum_schedule_per_sample(mm_per_sample, epoch_size=epoch_size)

# Instantiate the trainer object to drive the model training.
learner = C.learners.momentum_sgd(model.parameters, lr_schedule, mm_schedule)

# Configure trainer callbacks. This is the main point that this sample illustrates.
# Trainer callbacks are the mechanism via which logging, check-pointing, learning-rate
# adjustment, early stopping, and final testing are configured.

# Callback for progress logging loss and metric at the end of each epoch.
progress_writer = C.logging.ProgressPrinter()

# Callback for checkpointing. This will save a model every 'epoch_size' samples.
# Change 'restore' to True to have training start from a prior checkpoint file if available.
checkpoint_callback_config = C.CheckpointConfig(model_path, epoch_size, restore=False)

# Callback for cross-validation.
# The cross-validation callback mechanism allows you to implement your own
# learning-rate control and early stopping.
# The following implements a simple callback that halves the learning rate if the
# metric has not improved by at least 5% relative. The cross-validation callback
# gets configured to call this every 3*epoch_size samples, i.e. only every 3rd epoch.
prev_metric = 1 # metric from previous call to the callback. At very beginning, error rate is 100%.
def adjust_lr_callback(index, average_error, cv_num_samples, cv_num_minibatches):
    global prev_metric
    if (prev_metric - average_error) / prev_metric < 0.05: # relative gain must reduce metric by at least 5% rel
        learner.reset_learning_rate(C.learning_parameter_schedule_per_sample(learner.learning_rate() / 2))
        if learner.learning_rate() < lr_per_sample / (2**7-0.1): # we are done after the 6-th LR cut
            print("Learning rate {} too small. Training complete.".format(learner.learning_rate()))
            return False # means we are done
        print("Improvement of metric from {:.3f} to {:.3f} insufficient. Halving learning rate to {}.".format(prev_metric, average_error, learner.learning_rate()))
    prev_metric = average_error
    return True # means continue
cv_callback_config = C.CrossValidationConfig((X_cv, Y_cv), 3*epoch_size, minibatch_size=256,
                                             callback=adjust_lr_callback, criterion=criterion)

# Callback for testing the final model.
test_callback_config = C.TestConfig((X_test, Y_test), criterion=criterion)

# Configure distributed training.
# For this, we wrap the learner in a distributed_learner object.
# This specific example implements the BlockMomentum method. The Python script must be run
# using mpiexec in order to have effect. For example, under Windows, the command is:
#   mpiexec -n 4 -lines python -u MNIST_Complex_Training.py
learner = C.train.distributed.data_parallel_distributed_learner(learner)

# For distributed training, we must maximize the minibatch size, as to minimize
# communication cost and GPU underutilization. Hence, we use a "schedule"
# that increases the minibatch size after a few epochs. By specifying the learning rate
# as per sample, the contribution per sample maintains the same scale without
# having to fix up the learning rate.
# For this MNIST model, larger minibatch sizes make it faster, because the
# model is too small to utilize a full GPU. Hence data-parallel training cannot
# be expected to lead to speed-ups.
minibatch_size_schedule = C.minibatch_size_schedule([256]*6 + [512]*9 + [1024]*7 + [2048]*8 + [4096], epoch_size=epoch_size)

# Train and test, with checkpointing and learning-rate adjustment.
progress = criterion.train((X_train, Y_train), minibatch_size=minibatch_size_schedule,
                           max_epochs=50, parameter_learners=[learner],
                           callbacks=[progress_writer, checkpoint_callback_config, cv_callback_config, test_callback_config])

# Get progress statistics.
final_loss    = progress.epoch_summaries[-1].loss
final_metric  = progress.epoch_summaries[-1].metric
final_samples = progress.epoch_summaries[-1].samples
test_metric   = progress.test_summary.metric

# Inspect predictions on one minibatch, for illustration.
# For evaluation, we map the output of the network between 0-1 and convert them into probabilities
# for the two classes. We use a softmax function to get the probabilities of each of the class.
@C.Function.with_signature(C.layers.Tensor[input_shape])
def get_probability(data):
    return C.softmax(model(data))

X_check, Y_check = X_test[0:10000:400].copy(), Y_test[0:10000:400] # a small subsample of 25 examples
result = get_probability(X_check)

print("Label    :", [label.todense().argmax() for label in Y_check])
print("Predicted:", [result[i,:].argmax() for i in range(len(result))])

# Must call MPI finalize when process exit without exceptions
C.train.distributed.Communicator.finalize()
