# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root 
# for full license information.
# ==============================================================================

"""
MNIST Example, one hidden layer neural network
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from cntk import *

def add_dnn_sigmoid_layer(in_dim, out_dim, x, param_scale):
    W = LearnableParameter(out_dim, in_dim, initValueScale=param_scale)
    b = LearnableParameter(out_dim, 1, initValueScale=param_scale)
    t = Times(W, x)
    z = Plus(t, b)
    return Sigmoid(z)


def add_dnn_layer(in_dim, out_dim, x, param_scale):
    W = LearnableParameter(out_dim, in_dim, initValueScale=param_scale)
    b = LearnableParameter(out_dim, 1, initValueScale=param_scale)
    t = Times(W, x)
    return Plus(t, b)

if (__name__ == "__main__"):

    # Network definition
    feat_dim = 784
    label_dim = 10
    hidden_dim = 200

    training_filename = os.path.join("Data", "Train-28x28.txt")
    test_filename = os.path.join("Data", "Test-28x28.txt")

    features = Input(feat_dim, var_name='features')    
    f_reader = UCIFastReader(training_filename, 1, feat_dim)
    f_reader_t = UCIFastReader(test_filename, 1, feat_dim)
    
    feat_scale = Constant(0.00390625)
    feats_scaled = Scale(feat_scale, features)

    labels = Input(label_dim, tag='label', var_name='labels')
    l_reader = UCIFastReader(training_filename, 0, 1, label_dim, 
                             os.path.join("Data", "labelsmap.txt"))
    
    l_reader_t = UCIFastReader(test_filename, 0, 1, label_dim, 
                             os.path.join("Data", "labelsmap.txt"))
    
    h1 = add_dnn_sigmoid_layer(feat_dim, hidden_dim, feats_scaled, 1)
    out = add_dnn_layer(hidden_dim, label_dim, h1, 1)
    out.tag = 'output'

    ec = CrossEntropyWithSoftmax(labels, out)
    ec.tag = 'criterion'

    # Build the optimizer (settings are scaled down)
    my_sgd = SGDParams(epoch_size=600, minibatch_size=32,
                 learning_ratesPerMB=0.1, max_epochs=5, momentum_per_mb=0)

    # Create a context or re-use if already there
    with Context('mnist_one_layer' , clean_up=False) as ctx:
        # CNTK actions
        ctx.train(ec, my_sgd, {features:f_reader, labels:l_reader})
        ctx.infer({features:f_reader_t, labels:l_reader_t})
        print(ctx.test({features:f_reader_t, labels:l_reader_t}))
        
