# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from ..context import get_new_context, _CONTEXT
from ..graph import *
from ..ops.cntk2 import Abs, Plus, Minus, ElementTimes
from ..ops import constant, input_numpy, plus, times, past_value

import pytest

# keeping things short
A = np.asarray
C = constant
I = input_numpy


# testing whether operator overloads result in proper type
@pytest.mark.parametrize('root_node, expected', [
    # __add__ / __radd__
    (C(0) + C(1), Plus),
    (C(0) + 1, Plus),
    (0 + C(1), Plus),
    (0 + 1, int),

    # __sub__ / __rsub__
    (C(0) - C(1), Minus),
    (C(0) - 1, Minus),
    (0 - C(1), Minus),
    (0 - 1, int),

    # __mul__ / __rmul__ --> element-wise (!) multiplication
    (C(0) * C(1), ElementTimes),
    (C(0) * 1, ElementTimes),
    (0 * C(1), ElementTimes),
    (0 * 1, int),

    # __abs__
    (abs(C(0)), Abs),
])
def test_overload_types(root_node, expected):
    assert isinstance(root_node, expected)


def test_overload_exception():
    c = C(list(range(0, 10)))

    with pytest.raises(TypeError):
        c[:]

    with pytest.raises(TypeError):
        c[0:3:2]

def _to_list(desc):
    return [line.strip() for line in desc.split('\n')]


def test_graph_with_same_node_twice():
    v0 = constant(1)
    root_node = ops.plus(v0, v0)
    description, inputs = root_node._to_config_description()
    expected = ["v0 = ParameterTensor(1, learningRateMultiplier=0.0, init='fromLiteral', initValueScale=1, value=0, initFromFilePath='', initFromLiteral='1.0000", "', initOnCPUOnly=true, randomSeed=-1)",
                'v1 = CNTK2.Plus(v0, v0)']
    result = _to_list(description) 
    assert result == expected


if False:
    import scipy.sparse

    @pytest.mark.parametrize("alias, data, expected", [
        ('W', [A({})], ""),
        ('W', [{3: 1, 50: 1, 2: 0}, {1: -5}], """\
    0	|W 2:0 3:1 50:1
    1	|W 1:-5\
    """),
    ])
    def test_tensor_conversion_sparse(alias, data, expected):
        # We use the dictionary in data to create a SciPy sparse dictionary of
        # keys, which we then feed to the converter.
        dok_data = []
        for idx, data_elem in enumerate(data):
            d = scipy.sparse.dok_matrix((100, 1))
            for k, v in data_elem.items():
                d[k] = v
            dok_data.append(d)
        assert _tensor_to_text_format(idx, alias, dok_data) == expected


def test_loose_coupling():
    dh = past_value(1, 'outnode')
    out = times(dh, constant(2), name='outnode')

    expected = ["v0 = ParameterTensor(1, learningRateMultiplier=0.0, init='fromLiteral', initValueScale=1, value=0, initFromFilePath='', initFromLiteral='2.0000", "', initOnCPUOnly=true, randomSeed=-1)",
                'v1 = PastValue(1, outnode, timeStep=1, defaultHiddenActivation=0.1)',                
                'outnode = CNTK2.Times(v0, v1, outputRank=1)']

    description, inputs = out._to_config_description()
    assert _to_list(description) == expected
