# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import pytest
import numpy as np
import cntk
from cntk.contrib.crosstalkcaffe import utils
from cntk.contrib.crosstalkcaffe.unimodel import cntkmodel
from cntk.contrib.crosstalkcaffe.unimodel.cntkinstance import ApiSetup


def _layer_eq(layer, inputs, expected_out):
    out = layer(*inputs)
    assert (np.squeeze(np.array(out)) == np.squeeze(np.array(expected_out))).all()


def _layer_lt(layer, inputs, expected_out, eps=0.0001):
    out = layer(*inputs)
    assert (np.squeeze(np.array(out)) - np.squeeze(np.array(expected_out)) < eps).all()


def _install_test_layer(op_type, parameters, weights, input_data):
    para_cls_id = 'Cntk' + op_type + 'Parameters'
    para_instance = eval('.'.join(('cntkmodel', para_cls_id)))()
    for key, value in parameters.items():
        setattr(para_instance, key, value)
    layer_def = cntkmodel.CntkLayersDefinition()
    layer_def.parameters = para_instance
    layer_def.op_type = getattr(cntkmodel.CntkLayerType, utils.format.camel_to_snake(op_type))
    layer_def.op_name = '_'.join(('test', op_type))
    layer_def.parameter_tensor = []
    if weights is not None:
        for weight in weights:
            weight_tensor = cntkmodel.CntkTensorDefinition()
            weight_tensor.tensor = np.array(weight).shape
            weight_tensor.data = weight
            layer_def.parameter_tensor.append(weight_tensor)
    inputs_variable = []
    for input_tensor in input_data:
        inputs_variable.append(cntk.input(input_tensor.shape))
    return layer_def, inputs_variable

# TODO: It has been temporarily disabled due to a bug that causes any convolution test
#    (../../../layers/) to fail after op2cntk_test.py:test_conv_setup() is executed ALONG WITH
#    ../../deeprl/tests/policy_gradient_test.py:test_update_policy_and_value_function()
API_SETUP_CONV_DATA = [
    # The test case of conv ops
    (
        'Convolution',
        {
            'output': 2,
            'stride': [2, 2],
            'kernel': [3, 3],
            'auto_pad': False,
            'need_bias': True,
            'group': 1
        },
        [
            [[[[1., 2., 3.],
               [4., 5., 6.],
               [7., 8., 9.]]],
             [[[10., 11., 12.],
               [13., 14., 15.],
               [16., 17., 18.]]]],
            [[1., 2.]]
        ],
        [
            [[[1., 2., 3.],
              [4., 5., 6.],
              [7., 8., 9.]]]
        ],
        [
            [[[[286.]], [[692.]]]],
        ],
    ),
]


@pytest.mark.parametrize("op_type, parameters, weights, input_data, expected_out", API_SETUP_CONV_DATA)
def test_conv_setup(op_type, parameters, weights, input_data, expected_out):
    """
    The function to test conv api setup
    """
    pytest.skip('Temporarily skipping due to an issue with CrossTalkCaffe and Convolution')
    inputs = [np.array(item, dtype=np.float32) for item in input_data]
    outputs = [np.array(item, dtype=np.float32) for item in expected_out]
    layer_def, input_variants = _install_test_layer(op_type, parameters, weights, inputs)
    layer = getattr(ApiSetup, utils.format.camel_to_snake(op_type))(layer_def, input_variants)
    _layer_eq(layer, inputs, outputs)

# API_SETUP_POOLING_DATA = [
#     # The test cases of pool ops
#     (
#         'Pooling',
#         {
#             'stride': [2, 2],
#             'kernel': [2, 2],
#             'auto_pad': False,
#             'pooling_type': 0
#         },
#         [
#             [[[1., 2., 3., 4.],
#               [5., 6., 7., 8.],
#               [9., 10., 11., 12.],
#               [13., 14., 15., 16.]]]
#         ],
#         [
#             [[[[6., 8.],
#                [14., 16.]]]]
#         ]
#     ),
#     (
#         'Pooling',
#         {
#             'stride': [2, 2],
#             'kernel': [3, 3],
#             'auto_pad': True,
#             'pooling_type': 1
#         },
#         [
#             [[[1., 2., 3., 4.],
#               [5., 6., 7., 8.],
#               [9., 10., 11., 12.],
#               [13., 14., 15., 16.]]]
#         ],
#         [
#             [[[[3.5, 5., 6.],
#                [9.5, 11., 12.],
#                [13.5, 15, 16.]]]]
#         ]
#     )
# ]


# @pytest.mark.parametrize("op_type, parameters, input_data, expected_out", API_SETUP_POOLING_DATA)
# def test_pooling_setup(op_type, parameters, input_data, expected_out):
#     """
#     The function to test pooling api setup
#     """
#     inputs = [np.array(item, dtype=np.float32) for item in input_data]
#     outputs = [np.array(item, dtype=np.float32) for item in expected_out]
#     layer_def, input_variants = _install_test_layer(op_type, parameters, None, inputs)
#     layer = getattr(ApiSetup, utils.format.camel_to_snake(op_type))(layer_def, input_variants)
#     _layer_eq(layer, inputs, outputs)

API_SETUP_BN_DATA = [
    (
        'BatchNorm',
        {
            'epsilon': 0
        },
        [
            [[1., 1.]],
            [[2., 2.]],
            [1],
            [[0.5, 0.5]],
            [[1., 1.]]
        ],
        [
            [[[1., 2.],
              [3., 4.]],
             [[5., 6.],
              [7., 8.]]]
        ],
        [
            [[[[1., 1.353553],
               [1.707107, 2.06066]],
              [[2.414213, 2.76768],
               [3.12132, 3.474874]]]]
        ]
    )
]


@pytest.mark.parametrize("op_type, parameters, weights, input_data, expected_out", API_SETUP_BN_DATA)
def test_batch_norm_setup(op_type, parameters, weights, input_data, expected_out):
    pytest.skip('Temporarily skipping due to an issue with CrossTalkCaffe and Convolution')
    """
    The function to test batch norm api setup
    """
    inputs = [np.array(item, dtype=np.float32) for item in input_data]
    outputs = [np.array(item, dtype=np.float32) for item in expected_out]
    layer_def, input_variants = _install_test_layer(op_type, parameters, weights, inputs)
    layer = getattr(ApiSetup, utils.format.camel_to_snake(op_type))(layer_def, input_variants)
    _layer_lt(layer, inputs, outputs)

API_SETUP_DENSE_DATA = [
    (
        'Dense',
        {
            'num_output': 2,
        },
        [
            [[1.], [2.]],
            [[1.]]
        ],
        [
            [[[1, ]]]
        ],
        [
            [[[[2.], [3.]]]]
        ]
    )
]


@pytest.mark.parametrize("op_type, parameters, weights, input_data, expected_out", API_SETUP_DENSE_DATA)
def test_dense_setup(op_type, parameters, weights, input_data, expected_out):

    """
    The function to test dense api setup
    """
    inputs = [np.array(item, dtype=np.float32) for item in input_data]
    outputs = [np.array(item, dtype=np.float32) for item in expected_out]
    layer_def, input_variants = _install_test_layer(op_type, parameters, weights, inputs)
    layer = getattr(ApiSetup, utils.format.camel_to_snake(op_type))(layer_def, input_variants)
    _layer_eq(layer, inputs, outputs)

# API_SETUP_LRN_DATA = [
#     (
#         'LRN',
#         {
#             'kernel_size': 2,
#         },
#         [
#             [[[1., 2.]],
#              [[2., 3.]],
#              [[3., 4.]],
#              [[4., 5.]]]
#         ],
#         [
#             [[[[0.007416, 0.000463]],
#               [[0.000342, 0.000022]],
#               [[0.000022, 0.000002]],
#               [[0.000056, 0.000007]]]]
#         ]
#     )
# ]


# @pytest.mark.parametrize("op_type, parameters, input_data, expected_out", API_SETUP_LRN_DATA)
# def test_lrn_setup(op_type, parameters, input_data, expected_out):
#     """
#     The function to test dense api setup
#     """
#     inputs = [np.array(item, dtype=np.float32) for item in input_data]
#     outputs = [np.array(item, dtype=np.float32) for item in expected_out]
#     layer_def, input_variants = _install_test_layer(op_type, parameters, None, inputs)
#     layer = getattr(ApiSetup, utils.format.camel_to_snake(op_type))(layer_def, input_variants)
#     _layer_lt(layer, inputs, outputs)
