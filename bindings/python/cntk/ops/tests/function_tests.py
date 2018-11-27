# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for the function class.
"""

import numpy as np
import os
import pytest
import cntk as C
from .ops_test_utils import compare_lists_of_np_arrays, AA, cntk_device

from cntk.io import MinibatchSource, CTFDeserializer, StreamDefs, StreamDef


def test_variable_forwarding():
    op = C.constant(value=2, shape=(3,4)) + 1
    assert op.shape == (3,4)


def test_eval_by_node_name():
    i = C.input_variable(shape=(1,), needs_gradient=True, name='i')
    res = i + 3

    assert res.eval({i: [[3]]}) == [6]
    assert res.eval({'i': [[3]]}) == [6]
    assert res.eval({u'i': [[3]]}) == [6]


def test_replace_placeholders():
    p = C.placeholder(shape=(1,))
    i = C.input_variable(shape=(1,),
              needs_gradient=True,
              name='i')
    res = p + 3
    res.replace_placeholders({p: i})

    assert res.eval({i: [[3]]}) == [6]

    func = C.plus(i, 10)
    res2 = p + 3
    res2.replace_placeholders({p: func.output})

    assert res2.eval({i: [[3]]}) == [16]

    func = C.plus(i, 11)
    res3 = p + 3
    res3.replace_placeholders({p: func})

    assert res3.eval({i: [[3]]}) == [17]


def test_cloning():
    p = C.placeholder(shape=(1,), name='p')
    i = C.input_variable(shape=(1,),
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

    p = C.placeholder(shape=(1,2))
    c = C.constant(left_val)

    op = C.times(p, right_val)
    op.replace_placeholders({p:c})
    assert op.eval() == 26

    op = C.times(p, right_val)
    op.replace_placeholder(c)
    assert op.eval() == 26

def test_exception_for_unnamed_arguments():
    i1 = C.input_variable((1,2), name='i1')
    i2 = C.input_variable((2,1), name='i2')
    root_node = C.plus(i1, i2)
    input1 = [[[1,2]]]
    input2 = [[[[1],[2]]]]

    with pytest.raises(Exception):
        # not allowed, since plus has more than 1 input
        result = root_node.eval([input1, input2])

def test_output_in_intermediate_node():
    x = C.input_variable((2,))
    y = C.input_variable((2,))
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
    x = C.input_variable((2,))
    y = C.input_variable((2,))
    x0 = np.asarray([[2., 1.]])
    y0 = np.asarray([[4., 6.]])

    sum_node = x + 2

    # times_node not having sum_node
    times_node = x * y

    with pytest.raises(ValueError):
        sum_output = times_node.forward({x: x0, y: y0}, sum_node.outputs)


def test_evaluating_multiple_outputs():
    input_data = AA([1], np.float32)

    a = C.input_variable(shape=input_data.shape, name='a')
    a_plus_1 = a + 1
    out1 = ((a_plus_1 + 2) - 1) + 1
    out2 = ((a_plus_1 + 4) - 1) + 2
    z = C.combine([out1, out2])

    # create batch
    input_data.shape = (1, 1) + input_data.shape

    res = z.eval({a: input_data})

    expected_forward_out1 = [[4.]]
    expected_forward_out2 = [[7.]]
    assert np.array_equal(res[out1.output], expected_forward_out1)
    assert np.array_equal(res[out2.output], expected_forward_out2)

def test_set_name():
    x = C.input_variable((1,))
    y = C.input_variable((1,))
    x_plus_y = x + y
    assert (x_plus_y.name == '')
    x_plus_y.name = 'x_plus_y'
    assert (x_plus_y.name == 'x_plus_y')

    x_plus_y_2 = C.plus(x, y, name='x_plus_y_2')
    assert (x_plus_y_2.name == 'x_plus_y_2')
    with pytest.raises(ValueError):
        x_plus_y_2.name = 'x_plus_y_2_new'

    from ... import cntk_py
    cntk_py.allow_renaming_functions()

    x_plus_y_2.name = 'x_plus_y_2_new'


def test_data_type_inference():
    x_float = C.input_variable((1,), dtype = np.float64)
    param1 = C.parameter((C.InferredDimension, 1), init = C.glorot_uniform(), dtype = C.cntk_py.DataType_Unknown)
    assert (param1.get_data_type() == C.cntk_py.DataType_Unknown)

    x_times_param1 = C.times(x_float, param1)
    assert (param1.dtype == np.float64)

def test_recurrence_shape_inference():
    i = C.sequence.input_variable((2,))
    p = C.placeholder()
    p_past = C.sequence.past_value(p)
    p_past_plus_i = p_past + i

    p_past_plus_i.replace_placeholder(p_past_plus_i.output)
    assert p_past_plus_i.output.shape == (2,)

def test_sequence_data_mismatch():
    x = C.input_variable((1,), name='x')
    ones = C.sequence.input_variable((1,), name='ones')
    y_broadcast_last = C.sequence.broadcast_as(C.sequence.last(ones), x)
    y_broadcast_first = C.sequence.broadcast_as(C.sequence.first(ones), x)

    x0 = np.array([1,2,3,4],dtype=np.float32).reshape(4,1)
    o0 = np.array([1], dtype=np.float32).reshape(1,1)

    with pytest.raises(ValueError):
        y_broadcast_last_result = y_broadcast_last.eval({x:[x0], ones:[o0]})

    with pytest.raises(ValueError):
        y_broadcast_first_result = y_broadcast_first.eval({x:[x0], ones:[o0]})

def test_clone_with_function_in_substitution_map():
    input_dim = 1
    proj_dim = 2
    x = C.input_variable((input_dim,))
    w = C.parameter((input_dim, proj_dim))
    t = C.times(x, w)
    b = C.parameter((proj_dim))
    t_plus_b = t + b

    p = C.placeholder()
    just_b = t_plus_b.clone('clone', {t : p})
    t_plus_b_clone = just_b.clone('share', {p : t})

def test_clone_with_slice():
    i1 = C.input_variable((2,2), name='i1')
    i2 = C.input_variable((2,2), name='i2')
    x = C.splice(i1, i2, axis=0)
    W = C.constant(1, (4,1), name='W')
    y = C.convolution(W, x)
    assert(y.shape == (4,2))

    from ..functions import CloneMethod
    x1 = C.input_variable((2,1), name='x1')
    x2 = C.input_variable((2,1), name='x2')
    p1 = C.placeholder()
    p2 = C.placeholder()
    y_cloned = y.clone('clone', {i1:p1, i2:p2})
    y2 = y_cloned(x1, x2)
    assert(y2.shape == (4,1))

def test_as_composite():
    input_dim = 1
    proj_dim = 2
    x = C.input_variable((input_dim,))
    b = C.parameter((proj_dim))
    w = C.parameter((input_dim, proj_dim))
    func_name = 't_plus_b'
    t_plus_b = C.plus(C.times(x, w), b, name=func_name)
    assert(t_plus_b.root_function.name == func_name)
    composite = C.as_composite(t_plus_b.root_function)
    assert(composite.root_function.name == func_name)
    composite = C.as_composite(composite)
    assert(composite.root_function.name == func_name)
    composite = C.as_composite(t_plus_b)
    assert(composite.root_function.name == func_name)

def test_input_order():
    input_dim = 1
    proj_dim = 2
    x = C.input_variable((input_dim,), name='x')
    b = C.parameter((proj_dim), name='b')
    w = C.parameter((input_dim, proj_dim), name='w')
    func_name = 't_plus_b'
    t = C.times(x, w)
    t_plus_b = C.plus(t, b, name=func_name)

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
    x = C.input_variable((input_dim,), name='x')
    b = C.parameter((proj_dim), name='b')
    w = C.parameter((input_dim, proj_dim), name='w')
    func_name = 't_plus_b'
    t = C.times(x, w)
    t_plus_b = C.plus(t, b, name=func_name)

    duplicated_t_plus_b = C.combine([t_plus_b, t_plus_b])

    def compare_var_names(vars, names):
        num_vars = len(vars)
        for i in range(num_vars):
            if (vars[i].name != names[i]):
                return False

        return True

    assert compare_var_names(duplicated_t_plus_b.outputs, [func_name, func_name])


def test_extra_arguments_in_eval():
    x1 = C.input_variable((1,), name='x1')
    x2 = C.input_variable((1,), name='x2')
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

    f1 = C.input_variable(shape=(1,),
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
        gpu_device = C.gpu(0)
    except ValueError:
        pytest.skip('Test only runs when GPU available')

    device = cntk_device(device_id)
    x1 = C.input_variable(shape=())
    op1 = C.constant(value=1, shape=(1), device=device) + (C.constant(value=1, shape=(1), device=device) + x1)

    x2 = C.input_variable(shape=(1))

    # Deliberately locate the parameter on a different device
    # instead of the actual compute target device, so that
    # if we try to use this parameter, it results in an error
    if (device.type() == 0):
        parameter_device = gpu_device
    else:
        parameter_device = C.cpu()
    p = C.parameter(shape=(1), init=C.glorot_uniform(), device=parameter_device)
    op2 = (x2 - C.constant(value=10, shape=(1), device=device)) - p

    op = C.combine([op1, op2]);

    _, result = op.forward({x1 : np.asarray([1, 2, 3])}, [op1], device=device)
    assert np.array_equal(result[op1], np.asarray([[3], [4], [5]]))


def test_block_with_unused_outputs():
    p1 = C.placeholder()
    p3 = C.placeholder()
    func1 = C.as_block(p1 + 1, [(p1, p3)], 'plus_func_1')
    p2 = C.placeholder()
    p4 = C.placeholder()
    func2 = C.as_block(p2 + 1, [(p2, p4)], 'plus_func_2')
    p5 = C.placeholder()
    func3 = C.as_block(C.combine([func2]), [(p4, p5)], 'empty_block')
    input_var1 = C.input_variable(shape=())
    input_var2 = C.input_variable(shape=())
    block = C.as_block(C.combine([func1, func3]), [(p3, input_var1), (p5, input_var2)], 'multi_output_block')

    eval_root = C.combine([block.outputs[0]])
    result = eval_root.eval({input_var1 : np.asarray([3], dtype=np.float32), input_var2 : np.asarray([-3], dtype=np.float32)})
    assert np.array_equal(result, [ 4.])

def test_constant_data_type_mismatch():
    a = C.constant(np.triu(np.ones(5)), shape=(5,5))
    i = C.input_variable(shape=(5,5))
    b = a * i

    with pytest.raises(ValueError):
        b.eval({i:[[np.asarray(np.random.rand(5,5),dtype=np.float32)]]})

def test_update_signature():
    from cntk.layers.typing import Tensor

    input_dim = 14

    @C.Function
    def f(x):
        return x*x

    f.update_signature(Tensor[input_dim])

    assert f.outputs[0].shape == (input_dim,)
    assert f.x.shape == (input_dim,)


def test_eval_again_with_prev_outputs_live(device_id):
    x = C.input_variable(2)
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


def test_outputs_passing():
    in1 = C.input_variable(shape=(1,))
    a = C.alias(in1 + 1, name='a')
    b = a + 2

    expected = [[2], [3]]

    result = b.eval({in1: [[1], [2]]}, outputs=a.outputs)
    assert np.array_equal(result, expected)
    
    result = b.eval({in1: [[1], [2]]}, outputs=list(a.outputs))
    assert np.array_equal(result, expected)
    
    result = b.eval({in1: [[1], [2]]}, outputs=a.output)
    assert np.array_equal(result, expected)
    
    result = b.eval({in1: [[1], [2]]}, outputs=a)
    assert np.array_equal(result, expected)
    
    with pytest.raises(TypeError):
        b.eval({in1: [[1], [2]]}, outputs=[a.outputs])

def test_set_dropout_rate_attribute():
    from cntk import dropout, input; from math import pi;

    dropout_node = dropout(input(1), dropout_rate=0.3)
    key = 'dropoutRate'
    
    root = dropout_node.root_function
    assert np.isclose(root.attributes[key], 0.3)
    
    root.set_attribute(key, 0.4)
    assert np.isclose(root.attributes[key], 0.4)

    dropout_node.set_attribute(key, 0.777)
    assert np.isclose(root.attributes[key], 0.777)

    dropout_node.set_attribute(key, pi)
    assert np.isclose(root.attributes[key], pi)


def test_set_rng_seed_attribute():
    from cntk import random_sample, input;

    random_sample_node = random_sample(input(1), 1, True, seed=123)
    key = 'rngSeed'

    root = random_sample_node.root_function
    assert root.attributes[key] == 123
    
    root.set_attribute(key, 11530328594546889191)
    assert root.attributes[key] == 11530328594546889191

    random_sample_node.set_attribute(key, 2**31)
    assert root.attributes[key] == 2**31

    
def test_custom_attributes(tmpdir):
    root = 0 + C.input_variable(())
    assert not root.custom_attributes.keys()
    root.custom_attributes['cleared'] = 'none'
    assert 'none' == root.custom_attributes['cleared']
    # replace the custom attributes entirely, so 'cleared' is dropped
    root.custom_attributes = {'test':'abc', 'dict':{'a':1, 'b':2}, 'list':[1,2,3]}
    root.custom_attributes['test2'] = 'def'
    model_file = os.path.join(str(tmpdir), 'custom_attr.dnn')
    root.save(model_file)
    root2 = C.load_model(model_file)
    assert 'abc' == root2.custom_attributes['test']
    assert {'a':1, 'b':2} == root2.custom_attributes['dict']
    assert [1,2,3]==root2.custom_attributes['list']
    assert 'def' == root2.custom_attributes['test2']


def test_clone_with_different_dynamic_axes():
    q_axis = C.Axis('q')
    a_axis = C.Axis('a')
    question_input = C.sequence.input(shape=10, is_sparse=True, sequence_axis=q_axis)
    answer_input = C.sequence.input(shape=10, is_sparse=True, sequence_axis=a_axis)

    rnn = C.layers.Recurrence(C.layers.LSTM(5))(question_input)
    rnn_cloned = rnn.clone(C.CloneMethod.share, {question_input:answer_input})


def test_clone_with_deep_rnn_chaining():
    def seq_op_func(seqinp):
        l = seqinp
        r = C.sequence.future_value(l)
        r = C.expand_dims(r, -len(seqinp.shape) - 1)
        res = l + r
        return res

    def rnn_seq(features):
        step_func = C.layers.GRU(1)
        seq = C.layers.Recurrence(step_func)(features)
        return seq

    feat = C.sequence.input_variable((40,), name='sequence_inp')
    c1 = rnn_seq(feat)
    seq_op_res = seq_op_func(c1)
    net = rnn_seq(seq_op_res)
    cloned = net.clone('freeze')


def test_clone_with_unfound_new_node():
    x = C.input_variable(())
    y = C.combine(x * x, x + x)
    y0 = y[0]
    y1 = y[1]
    y0_new = C.plus(y0,0, name="test")
    X=C.logging.find_by_name(y0_new, 'QueryReply_y')
    
    with pytest.raises(AttributeError):
        y_clone = y.clone(C.CloneMethod.share, {y0:y0_new, y1:X})


def test_clone_with_unfound_previous_node():
    x = C.input_variable(())
    y = C.combine(x * x, x + x)
    y0 = y[0]
    y1 = y[1]
    y0_new = C.plus(y0,0, name="test")
    X=C.logging.find_by_name(y0_new, 'QueryReply_y')
    
    with pytest.raises(AttributeError):
        y_clone = y.clone(C.CloneMethod.share, {X:y0_new})


def test_clone_with_wrong_type_node():
    x = C.input_variable(())
    y = C.combine(x * x, x + x)
    y0 = y[0]
    y1 = y[1]
    y0_new = C.plus(y0,0, name="test")
    X=C.logging.find_by_name(y0_new, 'QueryReply_y')

    a = 5
    
    with pytest.raises(TypeError):
        y_clone = y.clone(C.CloneMethod.share, {y0:a})