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
from .ops_test_utils import unittest_helper, _test_unary_op, _test_binary_op, \
                            AA, precision, PRECISION_TO_TYPE, cntk_device
import cntk as C
from cntk import Value
from cntk.axis import Axis
from cntk.internal import sanitize_dtype_cntk
from .. import constant


RESHAPE_TEST_CASES = [
    #(input_shape, output_shape, expected_output_shape)
    ((2, 3),    (3, 2), (3, 2)),
    ((2, 3),    (6, 1), (6, 1)),
    ((2, 3),    (6, 1), (6, 1)),
    ((6, 1),    (2, 3), (2, 3)),
    ((2, 3, 5), (5, 6), (5, 6)),
    ((2, 3, 5), (C.InferredDimension, 6), (5, 6)),
    ((2, 3, 5), (5, C.InferredDimension), (5, 6)),
]

@pytest.mark.parametrize("input_shape, output_shape, expected_output_shape",
                         RESHAPE_TEST_CASES)
def test_op_reshape(input_shape, output_shape, expected_output_shape, device_id, precision):
    # Reshaping is just moving the input values to different indexes of the result tensor.
    # If we compute the gradients on the unmodified tensor, reshape would get 1 for all inputs
    # For testing the gradients we want to have different gradients for each input index otherwise we can't
    # test if they get wrongly permuted during test. To this end we multiply
    # the reshaping result with itself.
    dev = cntk_device(device_id)
    from .. import reshape, element_times, input

    num_tensor_elements = np.multiply.reduce(input_shape)
    input_tensor = np.arange(
        num_tensor_elements, dtype=PRECISION_TO_TYPE[precision]).reshape(input_shape)
    input_reshaped = input_tensor.reshape(expected_output_shape)

    a = input(shape=input_tensor.shape,
              dtype=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
              needs_gradient=True,
              name='a')

    a_reshaped = reshape(a, output_shape)

    const_input_reshaped = constant(input_reshaped, device=dev)
    input_op = element_times(a_reshaped, const_input_reshaped)

    expected_forward = [input_reshaped**2]
    expected_backward = {a: input_tensor}

    # create batch
    input_tensor.shape = (1,) + input_tensor.shape

    forward_input = {a: input_tensor}

    unittest_helper(input_op,
                    forward_input, expected_forward, expected_backward,
                    device_id=device_id, precision=precision)

RESHAPE_SUBSHAPE_TEST_CASES = [
    #(input_shape, replacement_shape, begin_axis, end_axis, expected_output_shape)
    ((2, 3),    (3, 2),                   0,                      Axis.new_leading_axis(),  (3, 2)),
    ((2, 3),    (1),                      0,                      0,                        (1, 2, 3)),
    ((2, 3),    (1, 1),                   Axis.new_leading_axis(),Axis.new_leading_axis(),  (2, 3, 1, 1)),
    ((2, 3, 5), (C.InferredDimension),    0,                      Axis(2),                  (6, 5)),
    ((2, 3, 5), (C.InferredDimension),    Axis(-3),               -1,                       (6, 5)),
    ((6, 5),    (2, C.InferredDimension), 0,                      1,                        (2, 3, 5)),
]

@pytest.mark.parametrize("input_shape, replacement_shape, begin_axis, end_axis, expected_output_shape", RESHAPE_SUBSHAPE_TEST_CASES)
def test_op_reshape_subshape(input_shape, replacement_shape, begin_axis, end_axis, expected_output_shape, device_id, precision):
    # Reshaping is just moving the input values to different indexes of the result tensor.
    # If we compute the gradients on the unmodified tensor, reshape would get 1 for all inputs
    # For testing the gradients we want to have different gradients for each input index otherwise we can't
    # test if they get wrongly permuted during test. To this end we multiply
    # the reshaping result with itself.
    dev = cntk_device(device_id)
    from cntk.internal import sanitize_dtype_cntk
    from .. import reshape, element_times, input

    num_tensor_elements = np.multiply.reduce(input_shape)
    input_tensor = np.arange(
        num_tensor_elements, dtype=PRECISION_TO_TYPE[precision]).reshape(input_shape)
    input_reshaped = input_tensor.reshape(expected_output_shape)

    a = input(shape=input_tensor.shape,
              dtype=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
              needs_gradient=True,
              name='a')

    a_reshaped = reshape(a, replacement_shape, begin_axis, end_axis)

    const_input_reshaped = constant(input_reshaped, device=dev)
    input_op = element_times(a_reshaped, const_input_reshaped)

    expected_forward = [input_reshaped**2]
    expected_backward = {a: input_tensor}

    # create batch
    input_tensor.shape = (1,) + input_tensor.shape

    forward_input = {a: input_tensor}

    unittest_helper(input_op,
                    forward_input, expected_forward, expected_backward,
                    device_id=device_id, precision=precision)


# Test that reshape accumulates the gradient in its input operand
# instead of overwriting the input operand gradient
def test_op_reshape_gradient_accumulation(device_id, precision):
    from .. import reshape, input

    input_shape = (2,3)
    output_shape = (3,2)
    expected_output_shape = (3,2)

    num_tensor_elements = np.multiply.reduce(input_shape)
    input_tensor = np.arange(
        num_tensor_elements, dtype=PRECISION_TO_TYPE[precision])
    input_reshaped = input_tensor.reshape(expected_output_shape)

    a = input(shape=input_tensor.shape,
              dtype=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
              needs_gradient=True,
              name='a')

    a_reshaped1 = reshape(a, output_shape)
    a_reshaped2 = reshape(a, output_shape)

    input_op = a_reshaped1 + a_reshaped2

    resulting_multiplicative_factor = 2
    expected_forward = [input_reshaped * resulting_multiplicative_factor]

    # create batch
    input_tensor.shape = (1,) + input_tensor.shape
    expected_backward = {a: np.full(input_tensor.shape, resulting_multiplicative_factor, dtype=PRECISION_TO_TYPE[precision])}

    forward_input = {a: input_tensor}

    unittest_helper(input_op,
                    forward_input, expected_forward, expected_backward,
                    device_id=device_id, precision=precision)


def test_op_reshape_parameter():
    from .. import reshape, parameter

    param_shape = (4,2)
    param_value = np.random.random(param_shape)
    param = parameter(init=param_value)
    param_new_shape = (8,1)
    param_reshaped = reshape(param, param_new_shape)

    expected_forward = np.copy(param_value).reshape(param_new_shape)
    state, result = param_reshaped.forward({}, [param_reshaped.output],
                                           [param_reshaped.output])
    assert np.allclose(result[param_reshaped.output], expected_forward)
    
    grad = param_reshaped.backward(state, np.ones(param_new_shape), [param])
    assert np.allclose(grad[param], np.ones(param_shape))


SLICE_TEST_CASES_STATIC = [
    #(input_data, slice_params(beg_index, end_index, axis), expected_result)
    ([[1, 2], [-3, 4]], (1, 2, 0), [[-3, 4]]),
    ([[1,2],[-3,4]], (1,2,1), [[2],[4]]),
]

@pytest.mark.parametrize("input_data, slice_params, expected_result",
                         SLICE_TEST_CASES_STATIC)
def test_op_slice(input_data, slice_params, expected_result, device_id, precision):

    input_data = AA(input_data, dtype=PRECISION_TO_TYPE[precision])

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

    expected_forward = AA([expected_result], dtype=PRECISION_TO_TYPE[precision])
    expected_backward = {
        'arg': [grad_slice(np.asarray(input_data), *slice_params)]
    }

    _test_unary_op(precision, device_id, C.slice, input_data,
                   expected_forward, expected_backward,
                   {'begin_index': slice_params[0],
                    'end_index': slice_params[1],
                    'axis': slice_params[2]})

SLICE_OVERLOAD_TEST_CASES_STATIC = [
    # (input_data, slices, axis, expected_result)

    ([[1, 2, 3], [-4, 5, 6]],
        # Selecting from row 1 the column 2
        (1, 2),
        [[6]]),

    # slicing with a list of indices
    ([[1, 2, 3], [-4, 5, 6]],
        # Selecting from both rows columns 1 and 2
        (0, [1, 2]),
        [[2, 3]]),
]


@pytest.mark.parametrize("input_data, slices, expected_result",
                         SLICE_OVERLOAD_TEST_CASES_STATIC)
def test_op_slice_overload(input_data, slices, expected_result,
                           device_id, precision):

    dtype = PRECISION_TO_TYPE[precision]
    input_data = AA(input_data, dtype=dtype)

    # Backward pass test
    # ==================
    # The gradient of the slice operator is a tensor of the same shape as the
    # input tensor, having 1 for elements that were taken and 0 for elements
    # that were dropped.

    def grad_slice(x, slices):
        res = np.zeros_like(x)
        res[slices] = 1
        return res

    value = AA(input_data, dtype=dtype)

    expected_forward = AA([expected_result], dtype=dtype)
    expected_backward = [grad_slice(input_data, slices)]

    a = C.input(shape=value.shape,
                dtype=sanitize_dtype_cntk(dtype),
                needs_gradient=True,
                name='a')

    f = a+0

    # create batch
    value.shape = (1,) + value.shape

    input_op = f[slices]

    forward_input = {a: value}
    expected_backward = {a: expected_backward}
    unittest_helper(input_op,
                    forward_input, expected_forward, expected_backward,
                    device_id=device_id, precision=precision)

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
        [[-4, 5, 6], [-14, 15, 16]]),
]


@pytest.mark.parametrize("input_data, slice_params, expected_result",
                         SLICE_TEST_CASES_DYNAMIC)
def test_op_slice_sequence(input_data, slice_params, expected_result,
                           device_id, precision):
    input_data = AA(input_data, dtype=PRECISION_TO_TYPE[precision])

    t = Axis.new_unique_dynamic_axis('t')
    sample_shape = input_data.shape[1:]
    a = C.sequence.input(shape=sample_shape,
                         dtype=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
                         needs_gradient=True,
                         sequence_axis=t,
                         name='a')

    result = C.sequence.slice(a, 
            begin_index=slice_params[0], 
            end_index=slice_params[1])

    def grad_slice(x, beg_index, end_index):
        res = np.zeros_like(x)
        res[beg_index:end_index] = 1
        return res


    expected_forward = AA([expected_result], 
            dtype=PRECISION_TO_TYPE[precision])
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
# contrib\Python\cntk\ops\tests\reshaping_test.py (check Git history)

SPLICE_TEST_CASES = [
    #(input_data1, input_data2, axis, expected_result)
    ([1], [2], 0, [1, 2]),
    ([1], [2], -1, [1, 2]),
    ([1], [2], Axis.new_leading_axis(), [[1], [2]]),
    ([1], [2], -2, [[1], [2]]),
    ([[1, 2], [4, 5]], [[10, 20], [30, 40], [50, 60]], 0,
     [[1, 2], [4, 5], [10, 20], [30, 40], [50, 60]]),
    ([[1, 2], [4, 5]], [[10, 20, 30], [40, 50, 60]], 1,
     [[1, 2, 10, 20, 30], [4, 5, 40, 50, 60]]),
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[10, 20], [30, 40]], 0,
     [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[10, 20], [30, 40]]]),
]


@pytest.mark.parametrize("input_data1, input_data2, axis, expected_result",
                         SPLICE_TEST_CASES)
def test_op_splice(input_data1, input_data2, axis, expected_result, device_id, precision):
    # FIXME This test currently fails in C++ with
    # RuntimeError: Node 'splice_ab' (RowStack operation): Attempted to
    # type-cast node to struct Microsoft::MSR::CNTK::INumInputs, which is not
    # possible.

    input_data1 = AA(input_data1, dtype=PRECISION_TO_TYPE[precision])
    input_data2 = AA(input_data2, dtype=PRECISION_TO_TYPE[precision])
    a = C.input(shape=input_data1.shape,
                dtype=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
                needs_gradient=True,
                name='a')
    b = C.input(shape=input_data2.shape,
                dtype=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
                needs_gradient=True,
                name='b')

    # create batch
    input_data1.shape = (1,) + input_data1.shape
    input_data2.shape = (1,) + input_data2.shape

    # splice using the operator
    root_op = C.splice(a, b, axis=axis, name='splice_ab')

    forward_input = {a: input_data1, b: input_data2}

    # Backward pass test
    # ==================
    # The gradient of the splice operator is all ones in the shape of the input

    def grad_splice(x):
        return np.ones_like(x)

    expected_forward = [expected_result]
    expected_backward = {
        a: grad_splice(np.asarray(input_data1)),
        b: grad_splice(np.asarray(input_data2))
    }

    unittest_helper(root_op,
                    forward_input, expected_forward, expected_backward,
                    device_id=device_id, precision=precision)


def test_op_gather_dynamic_axes_equivalence(device_id, precision):
    from .. import sequence

    input_data1 = AA([1], dtype=PRECISION_TO_TYPE[precision])
    input_data2 = AA([2], dtype=PRECISION_TO_TYPE[precision])

    a = sequence.input(shape=input_data1.shape,
                       dtype=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
                       name='a')
    b = sequence.input(shape=input_data2.shape,
                       dtype=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
                       name='b')

    is_last_a = sequence.is_last(a)
    a_last = sequence.gather(a, is_last_a)
    b_last = sequence.gather(b, is_last_a)
    z = a_last + b_last

    # create batch
    input_data1.shape = (1, 1) + input_data1.shape
    input_data2.shape = (1, 1) + input_data2.shape

    res = z.eval({a: input_data1, b: input_data2})
    expected_forward = [[[3.]]]
    assert np.array_equal(res, expected_forward)


def test_op_gather_derived_dynamic_axes_equivalence(device_id, precision):
    from .. import sequence

    input_data1 = AA([1], dtype=PRECISION_TO_TYPE[precision])
    input_data2 = AA([2], dtype=PRECISION_TO_TYPE[precision])

    a = sequence.input(shape=input_data1.shape,
                       dtype=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
                       name='a')
    b = sequence.input(shape=input_data2.shape,
                       dtype=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
                       name='b')

    a_last = sequence.gather(a, sequence.is_last(a), new_sequence_axis_typeinfo=(0, 1))
    b_last = sequence.gather(b, sequence.is_last(b), new_sequence_axis_typeinfo=(0, 1))

    z = a_last + b_last

    # create batch
    input_data1.shape = (1, 1) + input_data1.shape
    input_data2.shape = (1, 1) + input_data2.shape

    res = z.eval({a: input_data1, b: input_data2})
    expected_forward = [[3.]]
    assert np.array_equal(res, expected_forward)


def test_op_gather_sparse(device_id):
    from .. import sequence, times

    input_sparse_indices = [[1, 3, 5, 5], [2, 4], [0, 2]]
    vocab_size = 6
    input_data = Value.one_hot(input_sparse_indices, vocab_size)

    a = sequence.input(shape=(vocab_size,), is_sparse=True, name='a')

    a_last = sequence.last(a)
    a_last_dense = times(a_last, np.eye(vocab_size))
    res = a_last_dense.eval({a : input_data})
    assert np.array_equal(res, [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0]])

    a_last_2 = sequence.slice(a, -2, 0)
    a_last_2_dense = times(a_last_2, np.eye(vocab_size))
    res = a_last_2_dense.eval({a : input_data})
    assert np.array_equal(res, [[[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]], [[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]], [[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]])


def test_op_scatter_sparse(device_id):
    from .. import sequence, times

    input_sparse_indices = [[1, 3, 5, 5], [2, 4], [0, 2]]
    vocab_size = 6
    input_data = Value.one_hot(input_sparse_indices, vocab_size)

    a = sequence.input(shape=(vocab_size,), is_sparse=True, name='a')

    a_last_scatter = sequence.scatter(sequence.last(a), sequence.is_first(a))
    a_last_scatter_dense = times(a_last_scatter, np.eye(vocab_size))
    res = a_last_scatter_dense.eval({a : input_data})
    assert np.array_equal(res[0], np.asarray([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]))
    assert np.array_equal(res[1], np.asarray([[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0]]))
    assert np.array_equal(res[2], np.asarray([[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]]))


def test_op_broadcast_as(device_id, precision):
    from .. import sequence

    a_data = [AA([1], dtype=PRECISION_TO_TYPE[precision]), 
              AA([2], dtype=PRECISION_TO_TYPE[precision]), 
              AA([3], dtype=PRECISION_TO_TYPE[precision])]
    b_data = [AA([[2]], dtype=PRECISION_TO_TYPE[precision]), 
              AA([[2], [3]], dtype=PRECISION_TO_TYPE[precision]), 
              AA([[2], [3], [4]], dtype=PRECISION_TO_TYPE[precision])]

    a = C.input(shape=(1,), dtype=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]), name='a')
    b = sequence.input(shape=(1,), dtype=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]), name='b')

    broadcast_a_as_b = sequence.broadcast_as(a, b)
    
    res = broadcast_a_as_b.eval({a: a_data, b: b_data})
    assert np.array_equal(res[0], np.asarray([[1.]]))
    assert np.array_equal(res[1], np.asarray([[2.], [2.]]))
    assert np.array_equal(res[2], np.asarray([[3.], [3.], [3.]]))


def test_op_broadcast_as_in_loop(device_id):
    from .. import sequence, placeholder, input

    a_data = [AA([1]), AA([2]), AA([3])]
    b_data = [AA([[2]]), AA([[2], [3]]), AA([[2], [3], [4]])]

    a = C.input(shape=(1,), name='a')
    b = sequence.input(shape=(1,), name='b')

    out_placeholder = placeholder()
    out_delayed = sequence.past_value(out_placeholder, time_step=5)
    out_delayed_plus_b = out_delayed + b
    out = sequence.broadcast_as(a, out_delayed_plus_b)
    out.replace_placeholder(out)
    
    res = out.eval({a: a_data, b: b_data})
    assert np.array_equal(res[0], np.asarray([[1.]]))
    assert np.array_equal(res[1], np.asarray([[2.], [2.]]))
    assert np.array_equal(res[2], np.asarray([[3.], [3.], [3.]]))


def test_op_sequence_reduce_sum(device_id, precision):
    from .. import sequence

    a = sequence.input(shape=(1,), dtype=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]), needs_gradient=True, name='a')

    sequence_sum_a_plus_sequence_sum_a = sequence.reduce_sum(a) + sequence.reduce_sum(a)

    a_data = [AA([[2]], dtype=PRECISION_TO_TYPE[precision]),
              AA([[2], [3]], dtype=PRECISION_TO_TYPE[precision]),
              AA([[2], [3], [4]], dtype=PRECISION_TO_TYPE[precision])]

    actual_grad = sequence_sum_a_plus_sequence_sum_a.grad({a: a_data}, [a])
    assert np.array_equal(actual_grad[0], np.asarray([[2.]]))
    assert np.array_equal(actual_grad[1], np.asarray([[2.], [2.]]))
    assert np.array_equal(actual_grad[2], np.asarray([[2.], [2.], [2.]]))
    
    res = sequence_sum_a_plus_sequence_sum_a.eval({a: a_data})
    assert np.array_equal(res[0], np.asarray([4.]))
    assert np.array_equal(res[1], np.asarray([10.]))
    assert np.array_equal(res[2], np.asarray([18.]))

    # Verify that calling sequence reduction on a placeholder with known
    # shape but unknown dynamic axes does not result in a problem
    p = C.placeholder(shape=(1,))
    r = sequence.reduce_sum(p)
    r.replace_placeholder(a)
    
    res = r.eval({a: a_data})
    assert np.array_equal(res[0], np.asarray([2.]))
    assert np.array_equal(res[1], np.asarray([5.]))
    assert np.array_equal(res[2], np.asarray([9.]))
