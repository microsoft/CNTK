# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np

from cntk.ops import input, abs, square, sqrt, cos
from cntk.layers import For, Dense, SequentialClique, ResNetBlock, Sequential

import pytest

@pytest.mark.parametrize("layers_count, dense_units", [(4,5), (6,9), (7, 10)])
def test_for_constructor_layer(layers_count, dense_units):
    x = input(4)

    network = For(range(layers_count), lambda i: Dense(dense_units))

    expected_num_of_parameters = 2 * layers_count
    assert len(network.parameters) == expected_num_of_parameters

    res = network(x)

    expected_output_shape = (dense_units,)
    assert res.shape == expected_output_shape

def test_failing_for_constructor():
    with pytest.raises((ValueError, TypeError)):
        network = For(range(3), Dense(5))

    class MyFunction:
        def __call__(self, x):
            return Dense(x)

    with pytest.raises((ValueError, TypeError)):
        network = For(range(3), MyFunction())
    with pytest.raises((ValueError, TypeError)):
        network = For(range(3), MyFunction()(5))

INPUT_DATA = [[2, 8],[4, 7, 9], [5, 6, 10]]

@pytest.mark.parametrize("input_data", INPUT_DATA)
def test_sequential_clique_with_functions(input_data):
    x = input(len(input_data))

    seq_clique = SequentialClique([abs, sqrt, square])(x)

    assert seq_clique.shape == x.shape

    np_data = np.asarray(input_data, np.float32)
    res = seq_clique.eval(np_data)

    expected_res = np.abs(np_data) + np_data
    expected_res += np.sqrt(expected_res)
    expected_res = np.square(expected_res)

    expected_res.shape = (1,) + expected_res.shape

    np.testing.assert_array_almost_equal(res, expected_res, decimal=4)

@pytest.mark.parametrize("input_elements, expected", [(5,360.0), (7,1344.0)])
def test_sequential_clique_with_layers(input_elements, expected):
    x = input(input_elements)
    np_data = np.arange(input_elements, dtype=np.float32)

    unit_dense = Dense(input_elements, activation=None, init=1)

    seq_clique = SequentialClique([unit_dense, unit_dense, unit_dense])(x)

    assert seq_clique.shape == x.shape

    res = seq_clique.eval(np_data)

    assert res[0].shape == (input_elements,)
    assert np.unique(res[0])[0] == expected

@pytest.mark.parametrize("input_data", INPUT_DATA)
def test_sequential_constructor(input_data):
    x = input(len(input_data))
    np_data = np.asarray(input_data, np.float32)

    seq_layers = Sequential([abs, sqrt, square, cos])(x)

    assert seq_layers.shape == x.shape

    res = seq_layers(np_data)

    expected_res = np.cos(np.square(np.sqrt(np.abs(np_data))))

    np.testing.assert_array_almost_equal(res[0], expected_res, decimal=4)

@pytest.mark.parametrize("input_data", [[3, 5],[9, 25, 13]])
def test_resnet_block(input_data):
    x = input(len(input_data))

    res_net = ResNetBlock(square)(x)

    np_data = np.asarray(input_data, np.float32)

    actual_res = res_net.eval(np_data)

    expected_res = np.square(np_data) + np_data
    expected_res.shape = (1,) + expected_res.shape

    np.testing.assert_array_equal(actual_res, expected_res)

