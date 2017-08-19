# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for reshaping operations.
"""

from __future__ import division
import numpy as np
import cntk as C
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
    from .. import reshape, element_times

    num_tensor_elements = np.multiply.reduce(input_shape)
    input_tensor = np.arange(
        num_tensor_elements, dtype=PRECISION_TO_TYPE[precision]).reshape(input_shape)
    input_reshaped = input_tensor.reshape(expected_output_shape)

    a = C.input_variable(shape=input_tensor.shape,
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
    from .. import reshape, element_times

    num_tensor_elements = np.multiply.reduce(input_shape)
    input_tensor = np.arange(
        num_tensor_elements, dtype=PRECISION_TO_TYPE[precision]).reshape(input_shape)
    input_reshaped = input_tensor.reshape(expected_output_shape)

    a = C.input_variable(shape=input_tensor.shape,
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
    from .. import reshape

    input_shape = (2,3)
    output_shape = (3,2)
    expected_output_shape = (3,2)

    num_tensor_elements = np.multiply.reduce(input_shape)
    input_tensor = np.arange(
        num_tensor_elements, dtype=PRECISION_TO_TYPE[precision])
    input_reshaped = input_tensor.reshape(expected_output_shape)

    a = C.input_variable(shape=input_tensor.shape,
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
    ([[1, 2], [-3, 4]], (1, 2, 0, 1), [[-3, 4]]),
    ([[1,2],[-3,4]], (1,2,1, 1), [[2],[4]]),
    ([[1,2],[-3,4]], (0,2,1, -1), [[2, 1],[4, -3]]),
    ([[1,2],[-3,4]], (0,2,0, 2), [[1, 2]]),
	([[1,2],[-3,4], [-2,5], [7,8], [-9,6]], (0,5,0,2), [[1,2],[-2,5],[-9,6]]),
	([[1,2],[-3,4], [-2,5], [7,8], [-9,6]], (0,5,0,-2), [[-9,6],[-2,5],[1,2]])
]

@pytest.mark.parametrize("input_data, slice_params, expected_result",
                         SLICE_TEST_CASES_STATIC)
def test_op_slice(input_data, slice_params, expected_result, device_id, precision):

    input_data = AA(input_data, dtype=PRECISION_TO_TYPE[precision])

    def _ax_slices(x, beg_index, end_index, axis, strides):
        '''
        Creates a NumPy slicing array from slice operator's arguments
        '''
        ax_slices = []
        for i in range(0, len(x.shape)):
            if i == axis:
                if end_index >= x.shape[i]:
                    ax_slices.append(slice(beg_index, None, abs(strides)))
                else:
                    ax_slices.append(slice(beg_index, end_index, abs(strides)))
            else:
                ax_slices.append(slice(None))  # corresponds to ':'
        return ax_slices

    # Backward pass test
    # ==================
    # The gradient of the slice operator is a tensor of the same shape as the
    # input tensor, having 1 for elements that were taken and 0 for elements
    # that were dropped.

    def grad_slice(x, beg_index, end_index, axis, strides):
        res = np.zeros_like(x)
        ax_slices = _ax_slices(x, beg_index, end_index, axis, strides)
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
                    'axis': slice_params[2],
                    'strides': slice_params[3]})

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

    a = C.input_variable(shape=value.shape,
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
    a = C.sequence.input_variable(shape=sample_shape,
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

    def test_splice(shape1, shape2):
        a = C.input_variable(shape=shape1,
                    dtype=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
                    needs_gradient=True,
                    name='a')
        b = C.input_variable(shape=shape2,
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

    test_splice(input_data1.shape, input_data2.shape)
    # test with free dimension axis
    if axis is int and axis >= 0:
        input_shape1 = list(input_data1.shape)
        input_shape2 = list(input_data2.shape)

        input_shape1[axis] = C.FreeDimension
        input_shape2[axis] = C.FreeDimension
        test_splice(input_shape1, input_shape2)


def test_swapaxes_0d_1d_operands():
    x1 = C.input_variable(())
    with pytest.raises(ValueError):
        swapaxes_0d = C.swapaxes(x1)

    x2 = C.input_variable(2)
    with pytest.raises(ValueError):
        swapaxes_1d = C.swapaxes(x2)


def test_transpose():
    a = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
    from itertools import permutations
    for p in permutations(range(4)):
        assert np.array_equal(C.transpose(a, p).eval(), np.transpose(a, p))
    # test permutations over odd number of axes just in case
    b = a.reshape(6, 4, 5)
    for p in permutations(range(3)):
        assert np.array_equal(C.transpose(b, p).eval(), np.transpose(b, p))
    # test negative numbers
    for p in permutations(range(3)):
        q = [i - 3 for i in p]
        assert np.array_equal(C.transpose(b, q).eval(), np.transpose(b, q))

def test_transpose_backward():
    shape = (2, 3, 4)
    p = (2, 0, 1)
    x0 = np.arange(np.prod(shape), dtype=np.float32).reshape(*shape)
    shapet = tuple(shape[i] for i in p)
    x = C.input_variable(shape, needs_gradient=True)
    y = C.reduce_sum(C.cos(C.transpose(x, p)))
    xt = C.input_variable(shapet, needs_gradient=True)
    yt = C.reduce_sum(C.cos(xt))
    g = np.squeeze(y.grad({x:x0}))
    gt = np.squeeze(yt.grad({xt:np.transpose(x0, p)}))
    assert np.allclose(np.transpose(g, p), gt)


def test_op_reshape_free_dimension(device_id):
    dev = cntk_device(device_id)
    x = C.input_variable((C.FreeDimension, 2, 2))

    x_reshaped_1 = C.reshape(x, (-1,), 0, 2)
    data = [[[1, 2], [3, 4]]]
    result = x_reshaped_1.eval({x : np.asarray(data, dtype=np.float32)})
    assert np.array_equal(result[0], data[0])
    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    result = x_reshaped_1.eval({x : np.asarray(data, dtype=np.float32)})
    assert np.array_equal(result[0], np.reshape(data, (4, 2)))

    x_reshaped_2 = C.reshape(x, (-1,), 1, 3)
    data = [[[1, 2], [3, 4]]]
    result = x_reshaped_2.eval({x : np.asarray(data, dtype=np.float32)})
    assert np.array_equal(result[0], np.reshape(data, (1, 4)))
    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    result = x_reshaped_2.eval({x : np.asarray(data, dtype=np.float32)})
    assert np.array_equal(result[0], np.reshape(data, (2, 4)))

def test_gather_op(device_id, precision):
    a_data = [AA([[0],[1]], dtype=PRECISION_TO_TYPE[precision]),
              AA([[3],[4]], dtype=PRECISION_TO_TYPE[precision])]
    a = C.input_variable((2,1))
    r_data = np.arange(12).reshape(6,2).astype('f')
    r = C.parameter(shape=r_data.data, init=r_data)
    res = C.gather(r, a).eval({a:a_data})
    expectd = np.asarray([[[[0., 1.]],[[2., 3.]]],[[[6., 7.]],[[8.,9.]]]])
    assert np.array_equal(res, expectd)

    grads = C.gather(r, a).grad({a:a_data}, [r])
    expectd_grad = np.asarray([[1,1],[1,1],[0,0],[1,1],[1,1],[0,0]], dtype=np.float32)
    assert np.array_equal(grads, expectd_grad)
    
    b_data = [AA([[0,2],[1,3]], dtype=PRECISION_TO_TYPE[precision]),
              AA([[2,4],[3,5]], dtype=PRECISION_TO_TYPE[precision])]
    b = C.input_variable((2,2))
    res2 = C.gather(r, b).eval({b:b_data})

    expectd2 = np.asarray([[[[0., 1.],[4.,5.]],[[2., 3.],[6., 7.]]],[[[4., 5.],[8.,9.]],[[6., 7.], [10., 11.]]]])
    assert np.array_equal(res2, expectd2)

    #the following small model is to test the memory reuse issue of gather node.
    x = C.input((3, 4))
    x1 = C.to_sequence(x)
    w = C.parameter((5, 6), init=1)
    z = C.gather(w, x1)
    assert z.shape == (4, 6)
    #need the unpack node to trigger memory reuse.
    f = C.sequence.unpack(z, 0, no_mask_output=True)
    y = C.input((3, 4, 6))
    loss = C.reduce_mean(C.square(f - y), axis=-1)
    loss = C.reduce_mean(loss, axis=C.Axis.all_axes())

    g = C.constant(0, shape=w.shape)
    u = C.assign(w, g + 1)
    learner = C.cntk_py.universal_learner([w], [g], u)
    trainer = C.trainer.Trainer(loss, [loss], [learner])
    indices = np.asarray([[[1, 2, 1, 2]]])
    input = np.repeat(np.repeat(indices, 3, axis=1), 10, axis=0)
    lable = np.full((10, 3, 4, 6), 2)
    trainer.train_minibatch({x: input, y: lable})
    # the 2nd and 3rd rows should be udpated by gradients.
    assert np.mean(w.value[1, :]) < 1
    assert np.mean(w.value[2, :]) < 1
    # the other three rows should keep as 1
    assert np.isclose(np.mean(w.value[0, :]), 1)
    assert np.isclose(np.mean(w.value[3, :]), 1)
    assert np.isclose(np.mean(w.value[4, :]), 1)

def test_convert_dynamic_axis():
    #test fix batch size
    batch_size = 4
    a = C.constant(shape=(batch_size, 2, 3), value=1)
    dynamic_a = C.to_batch(a)
    assert len(dynamic_a.dynamic_axes) == 1
    assert dynamic_a.shape == (2, 3)

    x = C.input_variable((2, 3))
    y = x + dynamic_a

    const_a = C.unpack_batch(y)
    assert len(const_a.dynamic_axes) == 0
    assert const_a.shape == (C.FreeDimension, 2, 3)

    f = C.assign(a, const_a)
    z = x + 1
    data = np.arange(24).reshape(4, 2, 3).astype('f')
    f.eval({x:data})
    expected = z.eval({x:data})
    assert np.array_equal(a.value, expected)

    #test reshape with batch axis
    x = C.input_variable((2,3))
    const_x = C.unpack_batch(x)
    assert len(const_x.dynamic_axes) == 0
    assert const_x.shape == (C.FreeDimension, 2, 3)

    const_y = C.reshape(const_x, (-1, 3))
    assert const_y.shape == (C.FreeDimension, 3)
    y = C.to_batch(const_y)
    assert len(y.dynamic_axes) == 1
    assert y.shape == (3,)

    z = y * 2
    expected = data.reshape((8, 3)) * 2
    assert np.array_equal(z.eval({x:data}), expected)

    #test inferred dimension
    x = C.input_variable((C.InferredDimension, 3))
    const_x = C.unpack_batch(x)
    assert len(const_x.dynamic_axes) == 0
    assert const_x.shape == (C.FreeDimension, C.InferredDimension, 3)

    const_y = const_x * 2
    y = C.to_batch(const_y)
    assert len(y.dynamic_axes) == 1
    assert y.shape == (C.InferredDimension, 3)

def test_pad():
    x = C.constant(value=np.arange(6).reshape((2,3)))
    pad1 = C.pad(x, [(1, 1), (2, 2)]).eval()
    expect1 = np.lib.pad([[0, 1, 2], [3, 4, 5]], ((1, 1), (2, 2)), 'constant')
    assert np.array_equal(pad1, expect1)

    pad2 = C.pad(x, [(1, 1), (2, 2)], mode=1).eval()
    expect2 = np.lib.pad([[0, 1, 2], [3, 4, 5]], ((1, 1), (2, 2)), 'reflect')
    assert np.array_equal(pad2, expect2)

    pad3 = C.pad(x, [(1, 1), (2, 2)], mode=2).eval()
    expect3 = np.lib.pad([[0, 1, 2], [3, 4, 5]], ((1, 1), (2, 2)), 'symmetric')
    assert np.array_equal(pad3, expect3)

    #test inferred dimension and free dimension
    x = C.input((C.InferredDimension, 3))
    data = np.arange(12).reshape((2, 2, 3))
    pad4 = C.pad(x, [(1, 1), (2, 2)], mode=1).eval({x:data})
    expect4 = np.lib.pad([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]],
                         ((0,0),(1,1),(2,2)), 'reflect')
    assert np.array_equal(pad4, expect4)

    x = C.input((C.FreeDimension, 3))
    pad5 = C.pad(x, [(1, 1), (2, 2)], mode=2).eval({x: data})
    expect5 = np.lib.pad([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]],
                         ((0, 0), (1, 1), (2, 2)), 'symmetric')
    assert np.array_equal(pad5, expect5)

    #test grad
    x = C.parameter(init=np.arange(6).reshape((2,3)))
    p = C.pad(x, mode=C.ops.SYMMETRIC_PAD, pattern=[(1, 0), (2, 1)])
    grad = p.grad({}, [x])
    expect_grad = np.asarray([[4., 4., 4.],[2., 2., 2.]])
    assert np.array_equal(grad, expect_grad)

    p2 = C.pad(x, mode=C.ops.REFLECT_PAD, pattern=[(1, 1), (2, 2)])
    grad2 = p2.grad({}, [x])
    expect_grad2 = np.asarray([[4., 6., 4.], [4., 6., 4.]])
    assert np.array_equal(grad2, expect_grad2)

