# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
from ..context import *
from ..ops.cntk2 import Input
from ..sgd import *
from ..reader import *

def test_parse_shapes_1():
    output = '''\
FormNestedNetwork: WARNING: Was called twice for v3 Plus operation

Validating network. 5 nodes to process in pass 1.

Validating --> dummy_node = InputValue() :  -> [2 {1} x *]
Validating --> v0 = LearnableParameter() :  -> [4 x 1 {1,4}]
Validating --> v1 = Reshape (v0) : [4 x 1 {1,4}] -> [2 x 2 {1,2}]
Validating --> v2 = LearnableParameter() :  -> [1 x 1 {1,1}]
Validating --> v3 = Plus (v1, v2) : [2 x 2 {1,2}], [1 x 1 {1,1}] -> [2 x 2 {1,2}]

Validating network. 2 nodes to process in pass 2.


Validating network, final pass.



5 out of 5 nodes do not share the minibatch layout with the input data.

Post-processing network complete.
'''

    expected = {
        'dummy_node': (2, np.NaN),
        'v0': (4, 1),
        'v1': (2, 2),
        'v2': (1, 1),
        'v3': (2, 2)
    }

    assert LocalExecutionContext._parse_shapes_from_output(output) == expected


def test_parse_shapes_2():
    output = '''\
Validating --> v1 = LearnableParameter() :  -> [3 x 2 {1,3}]
Validating --> v2 = InputValue() :  -> [2 {1} x *]
Validating --> v3 = Times (v1, v2) : [3 x 2 {1,3}], [2 {1} x *] -> [3 {1} x *]
Validating --> v4 = LearnableParameter() :  -> [3 x 1 {1,3}]
Validating --> v5 = Plus (v3, v4) : [3 {1} x *], [3 x 1 {1,3}] -> [3 x 1 {1,3} x *]
'''

    expected = {
        'v1': (3, 2),
        'v2': (2, np.NaN),
        'v3': (3, np.NaN),
        'v4': (3, 1),
        'v5': (3, 1, np.NaN),
    }

    assert LocalExecutionContext._parse_shapes_from_output(output) == expected


def test_parse_eval_result_output_1():
    output = '''\
0	|w.shape 1 1
0	|w 60.000000
1	|w.shape 1 2
1	|w 22.000000
1	|w 24.000000'''
    list_of_tensors = LocalExecutionContext._parse_result_output(output)
    expected = [[[60]], [[22], [24]]]
    assert len(list_of_tensors) == len(expected)
    for res, exp in zip(list_of_tensors, expected):
        assert np.allclose(res, np.asarray(exp))


def test_parse_eval_result_output_2():
    output = '''\
0	|w.shape 8 1
0	|w 1.#IND -1.#IND 1.#INF00 -1.#INF nan -nan inf -inf 
'''
    data = LocalExecutionContext._parse_result_output(output)
    data = data[0][0]  # First sequence in first batch
    assert len(data) == 8
    # Windows
    assert np.isnan(data[0])
    assert np.isnan(data[1])
    assert np.isinf(data[2]) and data[2] > 0
    assert np.isinf(data[3]) and data[3] < 0
    # Linux
    assert np.isnan(data[4])
    assert np.isnan(data[5])
    assert np.isinf(data[6]) and data[6] > 0
    assert np.isinf(data[7]) and data[7] < 0


def test_parse_test_result_output():
    output = '''\
Final Results: Minibatch[1-1]: eval_node = 2.77790430 * 500; crit_node = 0.44370050 * 500; perplexity = 1.55846366
'''
    result = LocalExecutionContext._parse_test_result(output)

    assert result['perplexity'] == 1.55846366
    assert result['eval_node'] == 2.77790430
    assert result['crit_node'] == 0.44370050
    assert len(result) == 3
    
def test_export_deferred_context():
    X = Input(2)    
    reader = CNTKTextFormatReader("Data.txt")
    my_sgd = SGDParams()

    with DeferredExecutionContext() as ctx:
        input_map=reader.map(X, alias='I', dim=2)
        ctx.train(
            root_nodes=[X], 
            training_params=my_sgd,
            input_map=input_map)

        ctx.test(
                root_nodes=[X], 
                input_map=input_map)    

        ctx.write(input_map=input_map)    
        ctx.eval(X, input_map)  
        with open(ctx.export("name")) as config_file:
            assert config_file.readlines()[-1] == "command=Train:Test:Write:Eval"
        
