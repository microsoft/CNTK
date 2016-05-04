# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
MNIST Example, one hidden layer neural network using training and testing data 
generated through `uci_to_cntk_text_format_converter.py 
<https://github.com/Microsoft/CNTK/blob/master/Source/Readers/CNTKTextFormatReader/uci_to_cntk_text_format_converter.py>`_
to convert it to the CNTKTextFormatReader format.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import cntk as C


def add_dnn_sigmoid_layer(in_dim, out_dim, x, param_scale):
    W = C.parameter((out_dim, in_dim), init_value_scale=param_scale)
    b = C.parameter((out_dim, 1), init_value_scale=param_scale)
    t = C.times(W, x)
    z = C.plus(t, b)
    return C.sigmoid(z)


def add_dnn_layer(in_dim, out_dim, x, param_scale):
    W = C.parameter((out_dim, in_dim), init_value_scale=param_scale)
    b = C.parameter((out_dim, 1), init_value_scale=param_scale)
    t = C.times(W, x)
    return C.plus(t, b)

def train_eval_mnist_onelayer_from_file(criterion_name=None, eval_name=None):

    # Network definition
    feat_dim = 784
    label_dim = 10
    hidden_dim = 200
    
    cur_dir = os.path.dirname(__file__)

    training_filename = os.path.join(cur_dir, "Data", "Train-28x28_text.txt")
    test_filename = os.path.join(cur_dir, "Data", "Test-28x28_text.txt")

    features = C.input(feat_dim)
    features.name = 'features'

    feat_scale = C.constant(0.00390625)
    feats_scaled = C.element_times(features, feat_scale)

    labels = C.input(label_dim)
    labels.tag = 'label'
    labels.name = 'labels'

    traning_reader = C.CNTKTextFormatReader(training_filename)
    test_reader = C.CNTKTextFormatReader(test_filename)

    h1 = add_dnn_sigmoid_layer(feat_dim, hidden_dim, feats_scaled, 1)
    out = add_dnn_layer(hidden_dim, label_dim, h1, 1)
    out.tag = 'output'

    ec = C.cross_entropy_with_softmax(labels, out)
    ec.name = criterion_name
    ec.tag = 'criterion'
    
    eval = C.ops.square_error(labels, out)
    eval.name = eval_name
    eval.tag = 'eval'
    
    # Specify the training parameters (settings are scaled down)
    my_sgd = C.SGDParams(epoch_size=600, minibatch_size=32,
                       learning_rates_per_mb=0.1, max_epochs=5, momentum_per_mb=0)

    # Create a context or re-use if already there
    with C.LocalExecutionContext('mnist_one_layer', clean_up=True) as ctx:
        # CNTK actions
         ctx.train(
            root_nodes=[ec, eval],
            training_params=my_sgd,
            input_map=traning_reader.map(labels, alias='labels', dim=label_dim).map(features, alias='features', dim=feat_dim))
            
         result = ctx.test(
            root_nodes=[ec, eval],
            input_map=test_reader.map(labels, alias='labels', dim=label_dim).map(features, alias='features', dim=feat_dim))

         return result


def test_mnist_onelayer_from_file():
    result = train_eval_mnist_onelayer_from_file('crit_node', 'eval_node')

    TOLERANCE_ABSOLUTE = 1E-06
    assert result['SamplesSeen'] == 10000
    assert np.allclose(result['Perplexity'], 7.6323031, atol=TOLERANCE_ABSOLUTE)
    assert np.allclose(result['crit_node'], 2.0323896, atol=TOLERANCE_ABSOLUTE)
    assert np.allclose(result['eval_node'], 1.9882504, atol=TOLERANCE_ABSOLUTE)

if __name__ == "__main__":
    print(train_eval_mnist_onelayer_from_file('crit_node', 'eval_node'))
