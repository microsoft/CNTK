import numpy as np
from ..context import *


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

    assert Context._parse_shapes_from_output(output) == expected

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

    assert Context._parse_shapes_from_output(output) == expected

def test_parse_eval_result_output_1():
    output = '''\
0	|w.shape 1 1
0	|w 60.000000
1	|w.shape 1 2
1	|w 22.000000
1	|w 24.000000'''
    list_of_tensors = Context._parse_result_output(output)
    expected = [[[60]], [[22],[24]]]
    assert len(list_of_tensors) == len(expected)
    for res, exp in zip(list_of_tensors, expected):
        assert np.allclose(res, np.asarray(exp))


def test_parse_test_result_output():
    output = '''\
Final Results: Minibatch[1-1]: SamplesSeen = 500    v8: SquareError/Sample = 13.779223    v7: CrossEntropyWithSoftmax/Sample = 0.20016696    Perplexity = 1.2216067   ''' 
    result = Context._parse_test_result(output)

    assert result['SamplesSeen'] == 500
    assert result['Perplexity'] == 1.2216067
    assert result['v8'] == 13.779223
    assert result['v7'] == 0.20016696
    assert len(result) == 4
