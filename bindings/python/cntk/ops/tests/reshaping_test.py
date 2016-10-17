# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for reshaping operations. 
"""

from __future__ import division
import numpy as np
import pytest
from .ops_test_utils import unittest_helper, _test_unary_op, _test_binary_op, AA, I, precision, PRECISION_TO_TYPE
import cntk as C
from ...utils import sanitize_dtype_cntk

EPS_IN_LOG = 1e-37        # 1e-37 is the highest guaranteed precision
# the backward result returned by CNTK log() for epsilon
BACKWARD_RESULST_FOR_LOG_EPS = 9.08782e+36
LOG_OF_EPS_IN_LOG = -85.1  # log(EPS_IN_LOG)

RESHAPE_TEST_CASES = [
    #(input_shape, output_shape, expected_output_shape)
    ((2, 3),    (3, 2), (3, 2)),
    ((2, 3),    (6, 1), (6, 1)),
    ((2, 3),    (6, 1), (6, 1)),
    ((6, 1),    (2, 3), (2, 3)),
    ((2, 3, 5), (5, 6), (5, 6)),
    # now we test the feature that we can set one dimension of the output_shape to 0 meaning that it's value is inferred
    # FIXME 0 is for some reason not supported yet
    #((2, 3, 5), (0, 6), (5, 6)),
    #((2, 3, 5), (5, 0), (5, 6)),
]


@pytest.mark.parametrize("input_shape, output_shape, expected_output_shape", RESHAPE_TEST_CASES)
def test_op_reshape(input_shape, output_shape, expected_output_shape, device_id, precision):
    # Reshaping is just moving the input values to different indexes of the result tensor.
    # If we compute the gradients on the unmodified tensor, reshape would get 1 for all inputs
    # For testing the gradients we want to have different gradients for each input index otherwise we can't
    # test if they get wrongly permuted during test. To this end we multiply
    # the reshaping result with itself.

    from ...utils import sanitize_dtype_cntk
    from .. import reshape, element_times

    num_tensor_elements = np.multiply.reduce(input_shape)
    input_tensor = np.arange(
        num_tensor_elements, dtype=PRECISION_TO_TYPE[precision])
    input_reshaped = input_tensor.reshape(expected_output_shape)

    a = I(shape=input_tensor.shape,
          data_type=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
          needs_gradient=True,
          name='a')

    a_reshaped = reshape(a, output_shape)

    input_op = element_times(a_reshaped, input_reshaped)

    expected_forward = [[input_reshaped**2]]
    expected_backward = {a: input_tensor}

    # create batch
    input_tensor.shape = (1, 1) + input_tensor.shape

    forward_input = {a: input_tensor}

    unittest_helper(input_op,
                    forward_input, expected_forward, expected_backward,
                    device_id=device_id, precision=precision)


def test_op_reshape_bad_input():
    from .. import reshape

    a = I(shape=(4, 5))
    with pytest.raises(ValueError):
        reshape(a, (-1, 2, 3))


SLICE_TEST_CASES_STATIC = [
    #(input_data, slice_params(beg_index, end_index,axis), expected_result)
    ([[1, 2], [-3, 4]], (1, 2, 0), [[-3, 4]]),
    # FIXME slicing on axes >0 is not supported yet
    # ([[1,2],[-3,4]], (1,2,1), [[2],[4]]),
]


@pytest.mark.parametrize("input_data, slice_params, expected_result",
                         SLICE_TEST_CASES_STATIC)
def test_op_slice(input_data, slice_params, expected_result, device_id, precision):

    input_data = AA(input_data, dtype=PRECISION_TO_TYPE[precision])
    a = I(shape=input_data.shape,
          data_type=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
          needs_gradient=True,
          name='a')

    def _ax_slices(x, beg_index, end_index, axis):
        '''
        Creates a NumPy slicing array from slice operator's arguments
        '''
        ax_slices = []
        for i in range(0, len(x.shape)):
            if i == axis:
                if end_index >= x.shape[i]:
                    ax_slices.append([beg_index, ])
                else:
                    ax_slices.append([beg_index, end_index])
            else:
                ax_slices.append(slice(None))  # corresponds to ':'
        return ax_slices

    # slice using the overload
    if False:  # FIXME remove ones the overloads are in place
        # slice using the operator
        result = C.slice(a, *slice_params)
        ax_slices = _ax_slices(a, *slice_params)
        result = a[ax_slices]

        unittest_helper(result, None, [[expected_result]], device_id=device_id,
                        precision=precision, clean_up=True, backward_pass=False)

    # Backward pass test
    # ==================
    # The gradient of the slice operator is a tensor of the same shape as the
    # input tensor, having 1 for elements that were taken and 0 for elements
    # that were dropped.

    def grad_slice(x, beg_index, end_index, axis):
        res = np.zeros_like(x)
        ax_slices = _ax_slices(x, beg_index, end_index, axis)
        res[ax_slices] = x[ax_slices]
        res[res != 0] = 1
        return res

    expected_forward = [
        AA([expected_result], dtype=PRECISION_TO_TYPE[precision])]
    expected_backward = {
        'arg': [[grad_slice(np.asarray(input_data), *slice_params)]]
    }

    _test_unary_op(precision, device_id, C.slice, input_data,
                   expected_forward, expected_backward,
                   {'begin_index': slice_params[0],
                    'end_index': slice_params[1],
                    'axis': slice_params[2]})

SLICE_TEST_CASES_DYNAMIC = [
    #(input_data, slice_params(beg_index, end_index), expected_result)
    # Note that input_data contains sequences
    ([[[1, 2, 3]], [[-4, 5, 6]], [[7, 8, 9]]],
        (0, 2),
        [[[1, 2, 3]], [[-4, 5, 6]]]),
    ([[[1, 2, 3], [11, 12, 13]], [[-4, 5, 6], [-14, 15, 16]], [[7, 8, 9], [17, 18, 19]]],
        (0, 2),
        [[[1, 2, 3], [11, 12, 13]], [[-4, 5, 6], [-14, 15, 16]]]),
    ([[[1, 2, 3], [11, 12, 13]], [[-4, 5, 6], [-14, 15, 16]], [[7, 8, 9], [17, 18, 19]]],
        (1, 2),
        [[[-4, 5, 6], [-14, 15, 16]]]),
]


@pytest.mark.parametrize("input_data, slice_params, expected_result",
                         SLICE_TEST_CASES_DYNAMIC)
# FIXME enable once the ZeroesLike RuntimeError is fixed
def test_op_slice_sequence(input_data, slice_params, expected_result, device_id, precision):
    input_data = AA(input_data, dtype=PRECISION_TO_TYPE[precision])

    t = C.Axis.new_unique_dynamic_axis('t')
    sample_shape = input_data.shape[1:]
    a = I(shape=sample_shape,
          data_type=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
          needs_gradient=True,
          dynamic_axes=[C.Axis.default_batch_axis(), t],
          name='a')

    result = C.slice(a, axis=t, begin_index=slice_params[
                     0], end_index=slice_params[1])

    def grad_slice(x, beg_index, end_index):
        res = np.zeros_like(x)
        res[beg_index:end_index] = 1
        return res

    expected_gradient = grad_slice(np.asarray(input_data), *slice_params)

    expected_forward = AA(
        [expected_result], dtype=PRECISION_TO_TYPE[precision])
    expected_backward = {
        a: [grad_slice(np.asarray(input_data), *slice_params)]
    }

    # create batch
    input_data.shape = (1,) + input_data.shape

    forward_input = {a: input_data}
    unittest_helper(result,
                    forward_input, expected_forward, expected_backward,
                    device_id=device_id, precision=precision)

# FIXME once the overloads are in place, integrate test_op_slice_overload from
# F:\CNTKv2\contrib\Python\cntk\ops\tests\reshaping_test.py

SPLICE_TEST_CASES = [
    #(input_data1, input_data2, axis, expected_result)
    ([1], [2], 0, [1, 2]),
    ([[1, 2], [4, 5]], [[10, 20], [30, 40], [50, 60]], 0,
     [[1, 2], [4, 5], [10, 20], [30, 40], [50, 60]]),
    ([[1, 2], [4, 5]], [[10, 20, 30], [40, 50, 60]], 1,
     [[1, 2, 10, 20, 30], [4, 5, 40, 50, 60]]),
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[10, 20], [30, 40]], 0,
     [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[10, 20], [30, 40]]]),
]


@pytest.mark.parametrize("input_data1, input_data2, axis, expected_result", SPLICE_TEST_CASES)
def test_op_splice(input_data1, input_data2, axis, expected_result, device_id, precision):
    # FIXME This test currently fails in C++ with
    # RuntimeError: Node 'splice_ab' (RowStack operation): Attempted to
    # type-cast node to struct Microsoft::MSR::CNTK::INumInputs, which is not
    # possible.

    input_data1 = AA(input_data1, dtype=PRECISION_TO_TYPE[precision])
    input_data2 = AA(input_data2, dtype=PRECISION_TO_TYPE[precision])
    a = I(shape=input_data1.shape,
          data_type=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
          needs_gradient=True,
          name='a')
    b = I(shape=input_data2.shape,
          data_type=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
          needs_gradient=True,
          name='b')

    # create batch
    input_data1.shape = (1, 1) + input_data1.shape
    input_data2.shape = (1, 1) + input_data2.shape

    # splice using the operator
    root_op = C.splice((a, b), axis, name='splice_ab')

    forward_input = {a: input_data1, b: input_data2}

    # Backward pass test
    # ==================
    # The gradient of the splice operator is all ones in the shape of the input

    def grad_splice(x):
        return np.ones_like(x)

    expected_forward = [[expected_result]]
    expected_backward = {
        a: grad_splice(np.asarray(input_data1)),
        b: grad_splice(np.asarray(input_data2))
    }

    unittest_helper(root_op,
                    forward_input, expected_forward, expected_backward,
                    device_id=device_id, precision=precision)
