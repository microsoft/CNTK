# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for the function class.
"""

import numpy as np
import pytest
import cntk as C
from ..functions import *
from ...train.trainer import *
from ...initializer import glorot_uniform
from .. import constant, parameter, input, placeholder, times, plus, sequence, as_composite, combine, convolution, splice, as_block
from ... import InferredDimension, gpu, cpu
from .ops_test_utils import compare_lists_of_np_arrays, AA, cntk_device

from cntk.io import MinibatchSource, CTFDeserializer, StreamDefs, StreamDef

def test_variable_forwarding():
    op = constant(value=2, shape=(3,4)) + 1
    assert op.shape == (3,4)

def test_eval_by_node_name():
    i = input(shape=(1,), needs_gradient=True, name='i')
    res = i + 3

    assert res.eval({i: [[3]]}) == [6]
    assert res.eval({'i': [[3]]}) == [6]
    assert res.eval({u'i': [[3]]}) == [6]


def test_replace_placeholders():
    p = placeholder(shape=(1,))
    i = input(shape=(1,),
                       needs_gradient=True,
                       name='i')
    res = p + 3
    res.replace_placeholders({p: i})

    assert res.eval({i: [[3]]}) == [6]

    if False:
        res2 = p + 2
        from .. import plus
        func = plus(res2, 10)
        res2.replace_placeholders({p: func.output})

        assert res2.eval({i: [3]}) == [15]

def test_cloning():
    p = placeholder(shape=(1,), name='p')
    i = input(shape=(1,),
                       needs_gradient=True,
                       name='i')
    res = p + i

    with pytest.raises(ValueError):
        res.clone(2)

    from ..functions import CloneMethod

    # Test freeze
    cloned = res.clone(CloneMethod.freeze)
    assert cloned.inputs[0].name == 'p'
    assert cloned.inputs[0].uid != p.uid
    assert cloned.inputs[1].name == 'i'
    assert cloned.inputs[1].uid != i.uid

    cloned = res.clone('freeze')
    assert cloned.inputs[0].name == 'p'
    assert cloned.inputs[0].uid != p.uid
    assert cloned.inputs[1].name == 'i'
    assert cloned.inputs[1].uid != i.uid


def test_replace_placeholder_s():
    left_val = [[10,2]]
    right_val = [[2],[3]]

    p = placeholder(shape=(1,2))
    c = constant(left_val)

    op = times(p, right_val)
    op.replace_placeholders({p:c})
    assert op.eval() == 26

    op = times(p, right_val)
    op.replace_placeholder(c)
    assert op.eval() == 26

def test_exception_for_unnamed_arguments():
    i1 = input((1,2), name='i1')
    i2 = input((2,1), name='i2')
    root_node = plus(i1, i2)
    input1 = [[[1,2]]]
    input2 = [[[[1],[2]]]]

    with pytest.raises(Exception):
        # not allowed, since plus has more than 1 input
        result = root_node.eval([input1, input2])

def test_output_in_intermediate_node():
    x = input((2,))
    y = input((2,))
    x0 = np.asarray([[2., 1.]], np.float32)
    y0 = np.asarray([[4., 6.]], np.float32)

    sum_node = x + 2
    times_node = sum_node * y

    sum_output = times_node.forward({x: x0, y: y0}, sum_node.outputs)

    assert len(sum_output[1]) == 1
    assert np.allclose(list(sum_output[1].values())[0], np.asarray(x0 + 2))

    two_nodes_output = times_node.forward({x: x0, y: y0}, times_node.outputs + sum_node.outputs)

    assert len(two_nodes_output[1]) == 2

    sum_forward = np.asarray(x0 + 2)
    expected_results = [sum_forward, sum_forward * y0]

    assert compare_lists_of_np_arrays(list(two_nodes_output[1].values()), expected_results)

def test_getting_output_from_non_existent_node():
    x = input((2,))
    y = input((2,))
    x0 = np.asarray([[2., 1.]])
    y0 = np.asarray([[4., 6.]])

    sum_node = x + 2

    # times_node not having sum_node
    times_node = x * y

    with pytest.raises(ValueError):
        sum_output = times_node.forward({x: x0, y: y0}, sum_node.outputs)


def test_evaluating_multiple_outputs():
    input_data = AA([1], np.float32)

    a = input(shape=input_data.shape, name='a')
    a_plus_1 = a + 1
    out1 = ((a_plus_1 + 2) - 1) + 1
    out2 = ((a_plus_1 + 4) - 1) + 2
    z = combine([out1, out2])

    # create batch
    input_data.shape = (1, 1) + input_data.shape

    res = z.eval({a: input_data})
    print(res)

    expected_forward_out1 = [[4.]]
    expected_forward_out2 = [[7.]]
    assert np.array_equal(res[out1.output], expected_forward_out1)
    assert np.array_equal(res[out2.output], expected_forward_out2)

def test_set_name():
    x = input((1,))
    y = input((1,))
    x_plus_y = x + y
    assert (x_plus_y.name == '')
    x_plus_y.name = 'x_plus_y'
    assert (x_plus_y.name == 'x_plus_y')

    x_plus_y_2 = plus(x, y, name='x_plus_y_2')
    assert (x_plus_y_2.name == 'x_plus_y_2')
    with pytest.raises(ValueError):
        x_plus_y_2.name = 'x_plus_y_2_new'

    from ... import cntk_py
    cntk_py.allow_renaming_functions()

    x_plus_y_2.name = 'x_plus_y_2_new'


def test_data_type_inference():
    x_float = input((1,), dtype = np.float64)
    param1 = parameter((InferredDimension, 1), init = glorot_uniform(), dtype = cntk_py.DataType_Unknown)
    assert (param1.get_data_type() == cntk_py.DataType_Unknown)

    x_times_param1 = times(x_float, param1)
    assert (param1.dtype == np.float64)

def test_recurrence_shape_inference():
    i = sequence.input((2,))
    p = placeholder()
    p_past = sequence.past_value(p)
    p_past_plus_i = p_past + i

    p_past_plus_i.replace_placeholder(p_past_plus_i.output)
    assert p_past_plus_i.output.shape == (2,)

def test_sequence_data_mismatch():
    x = sequence.input((1,), name='x')
    ones = sequence.input((1,), name='ones')
    y_broadcast_last = sequence.broadcast_as(sequence.last(ones), x)
    y_broadcast_first = sequence.broadcast_as(sequence.first(ones), x)

    x0 = np.array([1,2,3,4],dtype=np.float32).reshape(4,1)
    o0 = np.array([1], dtype=np.float32).reshape(1,1)

    with pytest.raises(ValueError):
        y_broadcast_last_result = y_broadcast_last.eval({x:[x0], ones:[o0]})

    with pytest.raises(ValueError):
        y_broadcast_first_result = y_broadcast_first.eval({x:[x0], ones:[o0]})

def test_clone_with_function_in_substitution_map():
    input_dim = 1
    proj_dim = 2
    x = input((input_dim,))
    w = parameter((input_dim, proj_dim))
    t = times(x, w)
    b = parameter((proj_dim))
    t_plus_b = t + b

    p = placeholder()
    just_b = t_plus_b.clone('clone', {t : p})
    t_plus_b_clone = just_b.clone('share', {p : t})

def test_clone_with_slice():
    i1 = input((2,2), name='i1')
    i2 = input((2,2), name='i2')
    x = splice(i1, i2, axis=0)
    W = constant(1, (4,1), name='W')
    y = convolution(W, x)
    assert(y.shape == (4,2))

    from ..functions import CloneMethod
    x1 = input((2,1), name='x1')
    x2 = input((2,1), name='x2')
    p1 = placeholder()
    p2 = placeholder()
    y_cloned = y.clone('clone', {i1:p1, i2:p2})
    y2 = y_cloned(x1, x2)
    assert(y2.shape == (4,1))

def test_as_composite():
    input_dim = 1
    proj_dim = 2
    x = input((input_dim,))
    b = parameter((proj_dim))
    w = parameter((input_dim, proj_dim))
    func_name = 't_plus_b'
    t_plus_b = plus(times(x, w), b, name=func_name)
    assert(t_plus_b.root_function.name == func_name)
    composite = as_composite(t_plus_b.root_function)
    assert(composite.root_function.name == func_name)
    composite = as_composite(composite)
    assert(composite.root_function.name == func_name)
    composite = as_composite(t_plus_b)
    assert(composite.root_function.name == func_name)

def test_input_order():
    input_dim = 1
    proj_dim = 2
    x = input((input_dim,), name='x')
    b = parameter((proj_dim), name='b')
    w = parameter((input_dim, proj_dim), name='w')
    func_name = 't_plus_b'
    t = times(x, w)
    t_plus_b = plus(t, b, name=func_name)

    def compare_var_names(vars, names):
        num_vars = len(vars)
        for i in range(num_vars):
            if (vars[i].name != names[i]):
                return False

        return True

    assert compare_var_names(t.root_function.inputs, ['x', 'w'])
    assert compare_var_names(t.inputs, ['x', 'w'])
    assert compare_var_names(t_plus_b.inputs, ['x', 'w', 'b'])

def test_combine_duplicated_inputs():
    input_dim = 1
    proj_dim = 2
    x = input((input_dim,), name='x')
    b = parameter((proj_dim), name='b')
    w = parameter((input_dim, proj_dim), name='w')
    func_name = 't_plus_b'
    t = times(x, w)
    t_plus_b = plus(t, b, name=func_name)

    duplicated_t_plus_b = combine([t_plus_b, t_plus_b])

    def compare_var_names(vars, names):
        num_vars = len(vars)
        for i in range(num_vars):
            if (vars[i].name != names[i]):
                return False

        return True

    assert compare_var_names(duplicated_t_plus_b.outputs, [func_name, func_name])


def test_extra_arguments_in_eval():
    x1 = input((1,), name='x1')
    x2 = input((1,), name='x2')
    x1_plus_1 = x1 + 1
    x1_plus_1_plus_x2 = x1_plus_1 + x2

    result = x1_plus_1.eval({x1 : np.asarray([[1]]), x2 : np.asarray([[1]])})
    assert np.allclose(result, [[[2]]])


def test_MinibatchData_and_Value_as_input(tmpdir):

    mbdata = r'''0  |S0 100'''

    tmpfile = str(tmpdir/'mbtest.txt')
    with open(tmpfile, 'w') as f:
        f.write(mbdata)

    defs = StreamDefs(f1 = StreamDef(field='S0', shape=1))
    mb_source = MinibatchSource(CTFDeserializer(tmpfile, defs),
                                randomize=False)

    f1_si = mb_source.stream_info('f1')

    mb = mb_source.next_minibatch(1)

    f1 = input(shape=(1,),
                       needs_gradient=True,
                       name='f')
    res = f1 * 2

    assert res.eval({f1: mb[f1_si]}) == [[200]]
    # Test MinibatchData
    assert res.eval(mb[f1_si]) == [[200]]
    # Test Value
    assert res.eval(mb[f1_si].data) == [[200]]
    # Test NumPy (converted back from MinibatchData)
    assert res.eval(mb[f1_si].asarray()) == [[200]]
    # Test Value
    assert res.eval(mb[f1_si].data) == [[200]]


def test_output_subset_evaluation(device_id):
    try:
        gpu_device = gpu(0)
    except ValueError:
        pytest.skip('Test only runs when GPU available')

    device = cntk_device(device_id)
    x1 = input(shape=())
    op1 = constant(value=1, shape=(1), device=device) + (constant(value=1, shape=(1), device=device) + x1)

    x2 = input(shape=(1))

    # Deliberately locate the parameter on a different device
    # instead of the actual compute target device, so that
    # if we try to use this parameter, it results in an error
    if (device.type() == 0):
        parameter_device = gpu_device
    else:
        parameter_device = cpu()
    p = parameter(shape=(1), init=glorot_uniform(), device=parameter_device)
    op2 = (x2 - constant(value=10, shape=(1), device=device)) - p
    
    op = combine([op1, op2]);

    _, result = op.forward({x1 : np.asarray([1, 2, 3])}, [op1], device=device)
    assert np.array_equal(result[op1], np.asarray([[3], [4], [5]]))


def test_block_with_unused_outputs():
    p1 = placeholder()
    p3 = placeholder()
    func1 = as_block(p1 + 1, [(p1, p3)], 'plus_func_1')
    p2 = placeholder()
    p4 = placeholder()
    func2 = as_block(p2 + 1, [(p2, p4)], 'plus_func_2')
    p5 = placeholder()
    func3 = as_block(combine([func2]), [(p4, p5)], 'empty_block')
    input_var1 = input(shape=())
    input_var2 = input(shape=())
    block = as_block(combine([func1, func3]), [(p3, input_var1), (p5, input_var2)], 'multi_output_block')
    
    eval_root = combine([block.outputs[0]])
    result = eval_root.eval({input_var1 : np.asarray([3], dtype=np.float32), input_var2 : np.asarray([-3], dtype=np.float32)})
    assert np.array_equal(result, [[ 4.]])

def test_constant_data_type_mismatch():
    a = constant(np.triu(np.ones(5)), shape=(5,5))
    i = input(shape=(5,5))
    b = a * i

    with pytest.raises(ValueError):
        b.eval({i:[[np.asarray(np.random.rand(5,5),dtype=np.float32)]]})

def test_update_signature():
    from cntk.layers.typing import Tensor

    input_dim = 14

    @Function
    def f(x):
        return x*x

    f.update_signature(Tensor[input_dim])

    assert f.outputs[0].shape == (input_dim,)
    assert f.x.shape == (input_dim,)


def test_transpose_0d_1d_operands():
    x1 = C.input(())
    with pytest.raises(ValueError):
        transpose_0d = C.transpose(x1)

    x2 = C.input(2)
    with pytest.raises(ValueError):
        transpose_1d = C.transpose(x2)


def test_eval_again_with_prev_outputs_live(device_id):
    x = C.input(2)
    dev = cntk_device(device_id)
    w1 = C.parameter(init=np.asarray([1], dtype=np.float32), device=dev)
    w2 = C.parameter(init=np.asarray([-1], dtype=np.float32), device=dev)
    out1 = x + w1
    out2 = x + w2
    op = C.combine([out1, out2])

    result1 = op.eval({x : np.asarray([2, 5], dtype=np.float32)}, device=dev)
    assert np.array_equal(result1[out1.output], [[3, 6]])
    assert np.array_equal(result1[out2.output], [[1, 4]])

    result2 = op.eval({x : np.asarray([[-1, 4], [-4, 7]], dtype=np.float32)}, device=dev)
    assert np.array_equal(result2[out1.output], [[0, 5], [-3, 8]])
    assert np.array_equal(result2[out2.output], [[-2, 3], [-5, 6]])

    # result1 should still be valid
    assert np.array_equal(result1[out1.output], [[3, 6]])
    assert np.array_equal(result1[out2.output], [[1, 4]])

    result1 = op.eval({x : np.asarray([2, 5], dtype=np.float32)}, device=dev, as_numpy=False)
    assert np.array_equal(result1[out1.output].asarray(), [[3, 6]])
    assert np.array_equal(result1[out2.output].asarray(), [[1, 4]])

    result2 = op.eval({x : np.asarray([[-1, 4], [-4, 7]], dtype=np.float32)}, device=dev, as_numpy=False)
    assert np.array_equal(result2[out1.output].asarray(), [[0, 5], [-3, 8]])
    assert np.array_equal(result2[out2.output].asarray(), [[-2, 3], [-5, 6]])

    # Accessing result1 now will cause an error since it was a temporary that
    # is now erased, due to the subsequent eval call
    with pytest.raises(RuntimeError):
        assert np.array_equal(result1[out1.output].asarray(), [[3, 6]])
    
    grad_op = out1 + out2
    grad1 = grad_op.grad({x : np.asarray([2, 5], dtype=np.float32)}, wrt=[w1, w2], device=dev)
    assert np.array_equal(grad1[w1], [2])
    assert np.array_equal(grad1[w2], [2])

    grad2 = grad_op.grad({x : np.asarray([[-1, 4], [-4, 7]], dtype=np.float32)}, wrt=[w1, w2], device=dev)
    assert np.array_equal(grad2[w1], [4])
    assert np.array_equal(grad2[w2], [4])

    # grad1 should still be valid
    assert np.array_equal(grad1[w1], [2])
    assert np.array_equal(grad1[w2], [2])

    grad1 = grad_op.grad({x : np.asarray([2, 5], dtype=np.float32)}, wrt=[w1, w2], device=dev, as_numpy=False)
    assert np.array_equal(grad1[w1].asarray(), [2])
    assert np.array_equal(grad1[w2].asarray(), [2])

    grad2 = grad_op.grad({x : np.asarray([[-1, 4], [-4, 7]], dtype=np.float32)}, wrt=[w1, w2], device=dev, as_numpy=False)
    assert np.array_equal(grad2[w1].asarray(), [4])
    assert np.array_equal(grad2[w2].asarray(), [4])

    # Accessing grad1 now will cause an error since it was a temporary that
    # is now erased, due to the subsequent grad call
    with pytest.raises(RuntimeError):
        assert np.array_equal(grad1[w1].asarray(), [2])
