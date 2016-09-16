# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Example of logictic regression implementation using training and testing data
from a NumPy array. 
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import cntk as C


train_N = 1000
test_N = 500

# Mapping 2 numbers to 3 classes
feature_dim = 2
num_classes = 3

def synthetic_data(N, feature_dim, num_classes):
    # Create synthetic data using NumPy. 
    Y = np.random.randint(size=(N, 1), low=0, high=num_classes)

    # Make sure that the data is separable
    X = (np.random.randn(N, feature_dim)+3) * (Y+1)

    # converting class 0 into the vector "1 0 0", 
    # class 1 into vector "0 1 0", ...
    class_ind = [Y==class_number for class_number in range(num_classes)]
    Y = np.asarray(np.hstack(class_ind), dtype=int)

    return X, Y

def train_eval_logistic_regression_with_numpy(criterion_name=None,
        eval_name=None, device_id=-1):

    # for repro and tests :-)
    np.random.seed(1)

    train_X, train_y = synthetic_data(train_N, feature_dim, num_classes)
    test_X, test_y = synthetic_data(test_N, feature_dim, num_classes)

    # Set up the training data for CNTK. Before writing the CNTK configuration,
    # the data will be attached to X.reader.batch and y.reader.batch and then
    # serialized. 
    X = C.input_numpy(train_X)
    y = C.input_numpy(train_y)

    # define our network -- one weight tensor and a bias
    W = C.parameter(value=np.zeros(shape=(feature_dim, num_classes)))
    b = C.parameter(value=np.zeros(shape=(1, num_classes)))
    out = C.times(X, W) + b

    ce = C.cross_entropy_with_softmax(y, out)
    ce.tag = 'criterion'
    ce.name = criterion_name    
    
    eval = C.ops.cntk1.SquareError(y, out)
    eval.tag = 'eval'
    eval.name = eval_name

    my_sgd = C.SGDParams(epoch_size=0, minibatch_size=25,
            learning_rates_per_mb=0.1, max_epochs=3)

    with C.LocalExecutionContext('logreg_numpy', device_id=device_id, clean_up=True) as ctx:
        ctx.train(
                root_nodes=[ce,eval], 
                training_params=my_sgd)

        # For testing, we attach the test data to the input nodes.
        X.reader.batch, y.reader.batch = test_X, test_y
        result = ctx.test(root_nodes=[ce,eval])
        return result


def test_logistic_regression_with_numpy(device_id):
    result = train_eval_logistic_regression_with_numpy('crit_node',
            'eval_node', device_id)

    TOLERANCE_ABSOLUTE = 1E-06
    print(result)
    assert np.allclose(result['perplexity'], 2.33378225, atol=TOLERANCE_ABSOLUTE)
    assert np.allclose(result['crit_node'], 0.84749023, atol=TOLERANCE_ABSOLUTE)
    assert np.allclose(result['eval_node'], 2.69121655, atol=TOLERANCE_ABSOLUTE)

if __name__ == "__main__":
    print(train_eval_logistic_regression_with_numpy())
