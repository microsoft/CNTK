# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for function extension
"""

from __future__ import division, print_function
import numpy as np

from cntk import *
from cntk.learners import *
from cntk.ops import *
from cntk.ops.tests.ops_test_utils import cntk_device
from cntk.ops.functions import UserFunction

np.random.seed(0)

input_dim = 2
num_output_classes = 2


def generate_random_data_sample(sample_size, feature_dim, num_classes):
    # Create synthetic data using NumPy.
    Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

    # Make sure that the data is separable
    X = (np.random.randn(sample_size, feature_dim)+3) * (Y+1)
    X = X.astype(np.float32)
    class_ind = [Y==class_number for class_number in range(num_classes)]
    Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    return X, Y


def linear_layer(input_var, output_dim):
    input_dim = input_var.shape[0]
    times_param = parameter(shape=(input_dim, output_dim))
    bias_param = parameter(shape=(output_dim))

    t = times(input_var, times_param)
    return bias_param + t

def dense_layer(inp, output_dim, nonlinearity):
    r = linear_layer(inp, output_dim)
    r = nonlinearity(r)
    if isinstance(r, UserFunction):
        r = user_function(r)
    return r

def fully_connected_classifier_net(inp, num_output_classes, hidden_layer_dim,
                                   num_hidden_layers, nonlinearity):
    h = dense_layer(inp, hidden_layer_dim, nonlinearity)
    for i in range(1, num_hidden_layers):
        h = dense_layer(h, hidden_layer_dim, nonlinearity)
    r = linear_layer(h, num_output_classes)
    return r

def print_training_progress(trainer, mb, frequency):
    training_loss = "NA"
    eval_error = "NA"

    if mb%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average

    return mb, training_loss, eval_error

def train(nonlinearity, num_hidden_layers, device_id):
    from cntk.cntk_py import always_allow_setting_default_device
    always_allow_setting_default_device()
    try_set_default_device(cntk_device(device_id))
    np.random.seed(0)

    learning_rate = 0.5
    lr_schedule = learning_rate_schedule(learning_rate, UnitType.minibatch)

    mysamplesize = 64
    features, labels = generate_random_data_sample(mysamplesize, input_dim, num_output_classes)

    hidden_layers_dim = 50

    inp = input((input_dim), np.float32)
    label = input((num_output_classes), np.float32)

    z = fully_connected_classifier_net(inp, num_output_classes, hidden_layers_dim,
                                       num_hidden_layers, nonlinearity)

    loss = cross_entropy_with_softmax(z, label)
    eval_error = classification_error(z, label)

    learner = sgd(z.parameters, lr_schedule)
    trainer = Trainer(z, (loss, eval_error), [learner])


    minibatch_size = 25
    num_samples = 2500
    num_minibatches_to_train = num_samples / minibatch_size

    training_progress_output_freq = 20

    losses = []
    errors = []

    for i in range(0, int(num_minibatches_to_train)):
        features, labels = generate_random_data_sample(minibatch_size, input_dim, num_output_classes)

        # Specify the input variables mapping in the model to actual minibatch data for training
        trainer.train_minibatch({inp : features, label : labels},
                device=cntk_device(device_id))
        batchsize, loss, error = print_training_progress(trainer, i,
                                                         training_progress_output_freq)
        if not (loss == "NA" or error =="NA"):
            losses.append(loss)
            errors.append(error)

    return losses, errors


class MySigmoid(UserFunction):
    def __init__(self, arg, name='MySigmoid'):
        super(MySigmoid, self).__init__([arg], name=name)

    def forward(self, argument, device=None, outputs_to_retain=None):
        sigmoid_x = 1/(1+np.exp(-argument))

        return sigmoid_x, sigmoid_x

    def backward(self, state, root_gradients):
        sigmoid_x = state

        return root_gradients * sigmoid_x * (1 - sigmoid_x)

    def infer_outputs(self):
        return [output_variable(self.inputs[0].shape, self.inputs[0].dtype,
            self.inputs[0].dynamic_axes)]

def test_ext_user_sigmoid(device_id):
    np.random.seed(0)
    act_losses, act_errors = train(MySigmoid, 4, device_id)
    np.random.seed(0)
    exp_losses, exp_errors = train(sigmoid, 4, device_id)
    assert np.allclose(exp_losses, act_losses)
    assert np.allclose(exp_errors, act_errors)

def measure_runtime(device_id):
    import timeit
    np.random.seed(0)
    for num_hidden_layers in [1,2,4,8,16]:
        t = timeit.Timer('train(MySigmoid, %i, %s)'%(num_hidden_layers,
            device_id), setup="from __main__ import train, MySigmoid")
        timings_my_sigmoid = t.repeat(number=10)
        np.random.seed(0)
        t = timeit.Timer('train(sigmoid, %i, %s)'%(num_hidden_layers,
            device_id), setup="from __main__ import train, sigmoid")
        timings_sigmoid = t.repeat(number=10)

        print("%i\t%.2f\t%.2f"%(num_hidden_layers, min(timings_my_sigmoid), min(timings_sigmoid)))

if __name__=='__main__':
    print("CPU")
    measure_runtime(-1)
    print("GPU")
    measure_runtime(0)
