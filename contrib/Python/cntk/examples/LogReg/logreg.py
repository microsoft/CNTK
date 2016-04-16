# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root 
# for full license information.
# ==============================================================================

"""
Example of logictic regression implementation 
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from cntk import *

cur_dir = os.path.dirname(__file__)

# Using data from https://github.com/Microsoft/CNTK/wiki/Tutorial
train_file = os.path.join(cur_dir, "Train-3Classes.txt") 
test_file = os.path.join(cur_dir, "Test-3Classes.txt") 
mapping_file = os.path.join(cur_dir, "SimpleMapping-3Classes.txt")

def train_eval_logistic_regression(criterion_name=None, eval_name=None):
    X = Input(2)
    y = Input(3)
    
    # training data readers
    rx = UCIFastReader(train_file, 0, 2)
    ry = UCIFastReader(train_file, 2, 1, 2, mapping_file)
    
    # testing data readers
    rx_t = UCIFastReader(test_file, 0, 2)
    ry_t = UCIFastReader(test_file, 2, 1, 2, mapping_file)
    
    W = LearnableParameter(3, 2)
    b = LearnableParameter(3, 1)

    out = Times(W, X) + b
    out.tag = 'output'
    ce = CrossEntropyWithSoftmax(y, out)
    ce.var_name = criterion_name
    ce.tag = 'criterion'
    eval = SquareError(y, out)
    eval.tag = 'eval'
    eval.var_name = eval_name

    my_sgd = SGDParams(
        epoch_size=0, minibatch_size=25, learning_ratesPerMB=0.1, max_epochs=3)

    with Context('demo', clean_up=False) as ctx:

        ctx.train(root_nodes=[ce,eval], optimizer=my_sgd, input_reader={X:rx, y:ry})                
        result = ctx.test(input_reader={X:rx_t, y:ry_t})
        
        return result

def test_logistic_regression():
    result = train_eval_logistic_regression('crit_node', 'eval_node')

    assert result['SamplesSeen'] == 500
    assert np.allclose(result['Perplexity'], 1.2216067)
    assert np.allclose(result['crit_node'], 0.2001669)
    assert np.allclose(result['eval_node'], 27.558445)

if __name__ == "__main__":
    print(train_eval_logistic_regression())
