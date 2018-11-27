# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for the ability to leave some static axes of an input free
and inferred from the bound value during compute.
"""

import numpy as np
import pytest
import cntk as C

def test_free_static_dimension_basic():
    x = C.input_variable((C.FreeDimension, 2))
    w = C.parameter(init=np.asarray([[2, 5], [1, 3]], dtype=np.float32))
    t = C.times(x, w)

    x_data = np.asarray([[0.5, 0.2]], np.float32)
    w_grad, t_val = t.grad({x : x_data}, wrt=[w], outputs=[t])
    assert np.array_equal(t_val, np.asarray([[[1.2, 3.1]]], dtype=np.float32))
    assert np.array_equal(w_grad, np.asarray([[0.5, .5], [.2, .2]], dtype=np.float32))

    x_data = np.asarray([[0.5, 0.2], [0.1, .6]], np.float32)
    w_grad, t_val = t.grad({x : x_data}, wrt=[w], outputs=[t])
    assert np.allclose(t_val, np.asarray([[[1.2, 3.1], [0.8, 2.3]]], dtype=np.float32))
    assert np.array_equal(w_grad, np.asarray([[0.6, .6], [.8, .8]], dtype=np.float32))

    x_data = np.asarray([[0.5, 0.2]], np.float32)
    w_grad, t_val = t.grad({x : x_data}, wrt=[w], outputs=[t])
    assert np.array_equal(t_val, np.asarray([[[1.2, 3.1]]], dtype=np.float32))
    assert np.array_equal(w_grad, np.asarray([[0.5, .5], [.2, .2]], dtype=np.float32))


def test_free_static_axis_in_recurrence():
    x = C.sequence.input_variable((C.FreeDimension, 2))
    out_placeholder = C.placeholder()
    out_past = C.sequence.past_value(out_placeholder)
    wh = C.parameter(init=np.asarray([[2, 5], [1, 3]], dtype=np.float32))
    wx = C.parameter(init=np.asarray([[1, 4], [2, 5]], dtype=np.float32))
    out = C.times(x, wx) + C.times(out_past, wh)
    out.replace_placeholders({out_placeholder : out})
    
    x_data = np.asarray([[0.5, 0.2], [-0.7, 1.2]], np.float32)
    w_grad, out_val = out.grad({x : x_data}, wrt=[wh, wx], outputs=[out])
    assert np.allclose(out_val, [[[[0.9, 3.], [1.7, 3.2]]]])
    assert np.allclose(w_grad[wx], [[-0.2, -0.2], [1.4, 1.4]])


def test_reshape_free_static_axis():
    x = C.input_variable((C.FreeDimension, 2, 3))
    x_reshaped = C.reshape(x, (-1), 0, 2)
    assert x_reshaped.shape == (C.FreeDimension, 3)
    x_data = np.arange(12).reshape(2, 2, 3)
    result = x_reshaped.eval({x : x_data})
    assert np.array_equal(result[0], x_data.reshape(4, 3))

    x_data = np.arange(18).reshape(3, 2, 3)
    result = x_reshaped.eval({x : x_data})
    assert np.array_equal(result[0], x_data.reshape(6, 3))

    x_reshaped = C.reshape(x, (-1), 1, 3)
    assert x_reshaped.shape == (C.FreeDimension, 6)
    x_data = np.arange(12).reshape(2, 2, 3)
    result = x_reshaped.eval({x : x_data})
    assert np.array_equal(result[0], x_data.reshape(2, 6))

    x_reshaped = C.reshape(x, (4), 0, 2)
    assert x_reshaped.shape == (4, 3)
    x_data = np.arange(12).reshape(2, 2, 3)
    result = x_reshaped.eval({x : x_data})
    assert np.array_equal(result[0], x_data.reshape(4, 3))

    x_data = np.arange(6).reshape(1, 2, 3)
    with pytest.raises(ValueError):
        result = x_reshaped.eval({x : x_data})
   

def test_free_and_inferred_static_dimension():
    x = C.input_variable((C.FreeDimension, -1))
    w = C.parameter(init=np.asarray([[2, 5], [1, 3]], dtype=np.float32))
    t = C.times(x, w)

    x_data = np.asarray([[0.5, 0.2]], np.float32)
    w_grad, t_val = t.grad({x : x_data}, wrt=[w], outputs=[t])
    assert np.array_equal(t_val, np.asarray([[[1.2, 3.1]]], dtype=np.float32))
    assert np.array_equal(w_grad, np.asarray([[0.5, .5], [.2, .2]], dtype=np.float32))

    x_data = np.asarray([[0.5, 0.2], [0.1, .6]], np.float32)
    w_grad, t_val = t.grad({x : x_data}, wrt=[w], outputs=[t])
    assert np.allclose(t_val, np.asarray([[[1.2, 3.1], [0.8, 2.3]]], dtype=np.float32))
    assert np.array_equal(w_grad, np.asarray([[0.6, .6], [.8, .8]], dtype=np.float32))

    x_data = np.asarray([[0.5, 0.2]], np.float32)
    w_grad, t_val = t.grad({x : x_data}, wrt=[w], outputs=[t])
    assert np.array_equal(t_val, np.asarray([[[1.2, 3.1]]], dtype=np.float32))
    assert np.array_equal(w_grad, np.asarray([[0.5, .5], [.2, .2]], dtype=np.float32))

    x_data = np.asarray([[0.5, 0.2, 0.9]], np.float32)
    with pytest.raises(ValueError):
        w_grad, t_val = t.grad({x : x_data}, wrt=[w], outputs=[t])


def test_inferred_static_axis_in_recurrence():
    x = C.sequence.input_variable((-1, 2))
    out_placeholder = C.placeholder()
    out_past = C.sequence.past_value(out_placeholder)
    wh = C.parameter(init=np.asarray([[2, 5], [1, 3]], dtype=np.float32))
    wx = C.parameter(init=np.asarray([[1, 4], [2, 5]], dtype=np.float32))
    out = C.times(x, wx) + C.times(out_past, wh)
    out.replace_placeholders({out_placeholder : out})
    
    x_data = np.asarray([[0.5, 0.2], [-0.7, 1.2]], np.float32)
    w_grad, out_val = out.grad({x : x_data}, wrt=[wh, wx], outputs=[out])
    assert np.allclose(out_val, [[[[0.9, 3.], [1.7, 3.2]]]])
    assert np.allclose(w_grad[wx], [[-0.2, -0.2], [1.4, 1.4]])


def test_slice_with_inferred_static_axis():
    x = C.input_variable(shape=(C.InferredDimension, C.InferredDimension, 3))
    padding_shape = (3, C.InferredDimension, 3)
    y = C.splice(C.constant(value=0, shape=padding_shape), x, axis=0)
    assert y.shape == (-1, -1, 3)
    y = C.splice(x, C.constant(value=0, shape=padding_shape), axis=0)
    assert y.shape == (-1, -1, 3)


def test_free_dimension_broadcast():
    i0 = C.sequence.input_variable(shape=(5,))
    i0_unpacked, _ = C.sequence.unpack(i0, padding_value=0).outputs
    i1 = C.input_variable(shape=(5,))
    m = i0_unpacked * i1
    assert m.shape == (-3, 5)

    i1 = C.input_variable(shape=(1,5,))
    m = i0_unpacked * i1
    assert m.shape == (-3, 5)


def test_free_static_times():
    x = C.input_variable((C.FreeDimension, C.FreeDimension))
    w = C.parameter(init=np.asarray([[2, 5], [1, 3]], dtype=np.float32))
    t = C.times(x, w)

    x_data = np.asarray([[0.5, 0.2]], np.float32)
    w_grad, t_val = t.grad({x : x_data}, wrt=[w], outputs=[t])
    assert np.array_equal(t_val, np.asarray([[[1.2, 3.1]]], dtype=np.float32))
    assert np.array_equal(w_grad, np.asarray([[0.5, .5], [.2, .2]], dtype=np.float32))

    x_data = np.asarray([[0.5, 0.2], [0.1, .6]], np.float32)
    w_grad, t_val = t.grad({x : x_data}, wrt=[w], outputs=[t])
    assert np.allclose(t_val, np.asarray([[[1.2, 3.1], [0.8, 2.3]]], dtype=np.float32))
    assert np.array_equal(w_grad, np.asarray([[0.6, .6], [.8, .8]], dtype=np.float32))

    x_data = np.asarray([[0.5, 0.2]], np.float32)
    w_grad, t_val = t.grad({x : x_data}, wrt=[w], outputs=[t])
    assert np.array_equal(t_val, np.asarray([[[1.2, 3.1]]], dtype=np.float32))
    assert np.array_equal(w_grad, np.asarray([[0.5, .5], [.2, .2]], dtype=np.float32))


FREE_STATIC_AXIS_TIMES_FREE_STATIC_AXIS_DATA = [
    (1, (C.FreeDimension, C.FreeDimension), np.asarray([[[0.5, 0.2], [0.35, 0.75]]], np.float32),
     (C.FreeDimension, C.FreeDimension), np.asarray([[[0.7, 0.3], [0.7, 0.3]]], np.float32)),
    (1, (C.FreeDimension, C.FreeDimension, C.FreeDimension), np.asarray([[[[2.5, 4.5], [10.5, 12.5]]]], np.float32),
     (C.FreeDimension, C.FreeDimension, C.FreeDimension), np.asarray([[[[5.], [13.]], [[7.], [ 15.]]]], np.float32)),
    (2, (C.FreeDimension, C.FreeDimension, C.FreeDimension), np.asarray([[[[2.5, 4.5], [10.5, 12.5]]]], np.float32),
     (C.FreeDimension, C.FreeDimension, C.FreeDimension), np.asarray([[[[5.], [13.]], [[7.], [15.]]]], np.float32)),
]

@pytest.mark.parametrize("output_rank, x_input_shape, x_data, y_input_shape, y_data", FREE_STATIC_AXIS_TIMES_FREE_STATIC_AXIS_DATA)
def test_free_static_axis_times_free_static_axis(output_rank, x_input_shape, x_data, y_input_shape, y_data):
    x = C.input_variable(x_input_shape)
    y = C.input_variable(y_input_shape)
    t = C.times(x, y, output_rank=output_rank)
    cntk_result = t.eval({x: x_data, y: y_data})[0]
    np_result = []
    for x_item, y_item in zip(x_data, y_data): #zip over the batch axis
        item_res = np.tensordot(x_item, y_item, axes=len(x_item.shape) - output_rank)
        np_result.append(item_res)
    np_result = np.vstack(np_result)
    np.testing.assert_allclose(np_result, cntk_result)


FREE_UPPACK_AXIS_TIMES_UPPACK_AXIS_DATA = [
    (1, (2), np.asarray([[0.5, 0.2], [0.35, 0.75]], np.float32),
     (2), np.asarray([[0.7, 0.3], [0.7, 0.3]], np.float32)),
    (1, (2, 3), np.reshape(np.arange(3 * 2 * 3, dtype=np.float32), (-1, 2, 3)),
      (3, 2), np.reshape(np.arange(3 * 2 * 3, dtype=np.float32), (-1, 3, 2))),
    (1, (2, 3, 4), np.reshape(np.arange(3 * 2 * 3 * 4, dtype=np.float32), (-1, 2, 3, 4)),
     (4, 3, 2), np.reshape(np.arange(3 * 4 * 3 * 2, dtype=np.float32), (-1, 4, 3, 2))),
    (2, (2, 3, 4), np.reshape(np.arange(5 * 2 * 3 * 4, dtype=np.float32), (-1, 2, 3, 4)), #5 elements in batch axis
     (2, 4, 3), np.reshape(np.arange(5 * 2 * 4 * 3, dtype=np.float32), (-1, 2, 4, 3))), #5 elements in batch axis
]

@pytest.mark.parametrize("output_rank, x_input_shape, x_data, y_input_shape, y_data", FREE_UPPACK_AXIS_TIMES_UPPACK_AXIS_DATA)
def test_unpack_axis_times_transpose_unpack_axis(output_rank, x_input_shape, x_data, y_input_shape, y_data):
    #test free axis times from unpack batch
    x = C.input_variable(x_input_shape)
    y = C.input_variable(y_input_shape)
    xx = C.unpack_batch(x)
    yy = C.unpack_batch(y)
    yyy = C.transpose(yy, range(len(yy.shape))[::-1])
    t = C.times(xx, yyy, output_rank=output_rank)
    cntk_result = t.eval({x: x_data, y: y_data})
    np_result = np.tensordot(x_data, np.transpose(y_data), axes = len(x_data.shape) - output_rank)
    np.testing.assert_allclose(np_result, cntk_result)


FREE_STATIC_AXES_POOLING_DATA = [
    ((C.FreeDimension, C.FreeDimension, C.FreeDimension),
     C.AVG_POOLING,
     (2, 2),
     (2, 2),
     [[[[2.5, 4.5], [10.5, 12.5]]]]),
    ((C.FreeDimension, C.FreeDimension, C.FreeDimension),
     C.MAX_POOLING,
     (2, 2),
     (2, 2),
     [[[[5, 7], [13, 15]]]]),
    ((C.FreeDimension, C.FreeDimension, C.FreeDimension),
     C.AVG_POOLING,
     (2, 2),
     (2, 1),
     [[[[2.5, 3.5, 4.5], [10.5, 11.5, 12.5]]]])
]
@pytest.mark.parametrize("input_shape, pooling_type, window_shape, strides, expected", FREE_STATIC_AXES_POOLING_DATA)
def test_free_static_pooling(input_shape, pooling_type, window_shape, strides, expected):
    img = np.reshape(np.arange(16, dtype=np.float32), [1, 4, 4])
    x = C.input_variable(input_shape)
    avg_pooling = C.pooling(x, pooling_type, window_shape, strides)
    assert avg_pooling.shape == (C.FreeDimension, C.FreeDimension, C.FreeDimension)
    assert np.allclose(avg_pooling.eval({x:[img]}), np.asarray(expected, dtype=np.float32))


from cntk.ops.functions import Function, UserFunction
from .ops_test_utils import AA

class MultiFreeDimensionOutputUserFunction(UserFunction):
    def __init__(self, arg1, arg2, name='f1'):
        super(MultiFreeDimensionOutputUserFunction, self).__init__([arg1, arg2], name=name)

    def infer_outputs(self):
        return [C.output_variable(C.FreeDimension, self.inputs[0].dtype, self.inputs[0].dynamic_axes),
                C.output_variable(C.FreeDimension, self.inputs[0].dtype, self.inputs[0].dynamic_axes)]

    def forward(self, arguments, outputs, device=None, outputs_to_retain=None):
        assert len(self.inputs) == 2

        outputs[self.outputs[0]] = [a0 + 2 * a1 for a0, a1 in zip(*arguments)]
        outputs[self.outputs[1]] = [2 * a0 + a1 for a0, a1 in zip(*arguments)]

        return None

    def backward(self, state, root_gradients, variables):
        if self.inputs[0] in variables:
            variables[self.inputs[0]] = [r0 + 2 * r1 for r0, r1 in zip(root_gradients[self.outputs[0]], root_gradients[self.outputs[1]])]

        if self.inputs[1] in variables:
            variables[self.inputs[1]] = [2 * r0 + r1 for r0, r1 in zip(root_gradients[self.outputs[0]], root_gradients[self.outputs[1]])]

def test_multi_freedim_output_udf():
    dim = 2
    x = C.sequence.input_variable(dim, needs_gradient=True, name='x')
    y = C.sequence.input_variable(dim, needs_gradient=True, name='y')
    op = C.user_function(MultiFreeDimensionOutputUserFunction(x, y))

    x_data = [AA([[1., 2.], [3., 4.]], dtype=np.float32)]
    y_data = [AA([[5., 6.], [7., 8.]], dtype=np.float32)]
    result = op.eval({x: x_data, y: y_data})
    assert np.allclose(result[op.outputs[0]], x_data[0] + 2 * y_data[0])
    assert np.allclose(result[op.outputs[1]], 2 * x_data[0] + y_data[0])

    op = op.outputs[0] + op.outputs[1]
    gradients = op.grad({x: x_data, y: y_data}, op.arguments)
    assert np.allclose(gradients[op.arguments[0]], [[[3., 3.], [3., 3.]]])
    assert np.allclose(gradients[op.arguments[1]], [[[3., 3.], [3., 3.]]])
