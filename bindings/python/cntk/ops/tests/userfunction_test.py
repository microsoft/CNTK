# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for function extension
"""

from __future__ import division, print_function
import numpy as np
import pytest

from cntk import *
from cntk.trainer import *
from cntk.learner import *
from cntk.ops.functions import UserFunction
from .ops_test_utils import AA

class MyPlus(UserFunction):
    def __init__(self, arg1, arg2, name='f1'):
        super(MyPlus, self).__init__([arg1, arg2], name=name)

        self.forward_calls = 0
        self.backward_calls = 0

    def infer_outputs(self):
        return [output_variable(self.inputs[0].shape,
            self.inputs[0].dtype, self.inputs[0].dynamic_axes)]

    def clone(self, cloned_inputs):
        return MyPlus(cloned_inputs[0], cloned_inputs[1])

    def forward(self, arguments, device=None, outputs_to_retain=None):
        assert len(self.inputs)==2

        result = arguments[0] + arguments[1]

        self.forward_calls += 1

        return None, result

    def backward(self, state, root_gradients):
        self.backward_calls += 1

        return root_gradients

def test_ext_eval_1():
    dim = 4
    p = parameter(shape=(dim,), init=10, name='p')
    i = input_variable(dim, needs_gradient=True, name='i_var')
    m = user_function(MyPlus(i, constant(3)))
    z = m+p

    input_data = np.random.rand(dim)
    result = z.eval([input_data])
    assert np.allclose(result[0][0], input_data+3+10)

def test_ext_eval_2_only_param():
    dim = 4
    p = parameter(shape=(dim,), init=10, name='p')
    i = input_variable(dim, needs_gradient=True, name='i_var')
    m = user_function(MyPlus(p, constant(3)))
    # combine does not work
    # z = combine([m.output])
    z = m+i

    input_data = np.random.rand(dim)
    result = z.eval([input_data])
    assert np.allclose(result[0][0], input_data+3+10)

def test_ext_eval_3_no_input():
    dim = 4
    p = parameter(shape=(dim,), init=10, name='p')
    m = user_function(MyPlus(p, constant(3)))
    z = m+0

    result = z.eval()
    # No batch dimension since we have no input
    assert np.allclose(result, np.zeros_like(p)+10+3)

def test_ext_eval_4_a_inside_graph():
    dim = 4
    p_init = 10
    p = parameter(shape=(dim,), init=p_init, name='p')
    m = user_function(MyPlus(p, constant(3)))
    z = p * m

    result = z.eval()
    # No batch dimension since we have no input
    assert np.allclose(result, ((p_init*np.ones_like(result))+3)*p_init)

def test_ext_eval_4_b_inside_graph():
    dim = 4
    p_init = 10
    p = parameter(shape=(dim,), init=p_init, name='p')
    z = user_function(p * MyPlus(p, constant(3)))

    result = z.eval()
    # No batch dimension since we have no input
    assert np.allclose(result, ((p_init*np.ones_like(result))+3)*p_init)

def test_ext_eval_5_times():
    dim = 2
    p_init = 10
    p = parameter(shape=(dim,), init=p_init, name='p')
    m = user_function(MyPlus(p, constant(3)))
    z = times(m, parameter(shape=(2,50), init=2))

    result = z.eval()
    # No batch dimension since we have no input
    assert np.allclose(result, ((p_init*np.ones_like(result))+3)*2*2)

def test_ext_eval_6_clone():
    dim = 4
    i = input_variable(dim, needs_gradient=True, name='i_var')
    m = i + 3

    p = parameter(shape=(dim,), init=10, name='p')
    z = m + p
    
    m_udf = user_function(MyPlus(i, constant(3)))
    z_clone = z.clone('share', {m : m_udf} );

    input_data = np.random.rand(dim)
    result = z_clone.eval([input_data])
    assert np.allclose(result[0][0], input_data+3+10)

def test_ext_eval_7_placeholder():
    dim = 4
    p = parameter(shape=(dim,), init=10, name='p')
    i = input_variable(dim, needs_gradient=True, name='i_var')
    pl = placeholder_variable()
    m = user_function(MyPlus(pl, constant(3)))
    z = m+p
    z.replace_placeholder(i)

    input_data = np.random.rand(dim)
    result = z.eval([input_data])
    assert np.allclose(result[0][0], input_data+3+10)

def test_ext_train():
    dim = 4

    p = parameter(shape=(dim,), init=10)
    i = input_variable(dim, needs_gradient=True, name='i_var')
    m = MyPlus(i, constant(3))
    # keeping m unwrapped since we need to access its member variables
    z = user_function(m)+p

    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    lr_per_sample = learning_rate_schedule(0.007, UnitType.sample)
    trainer = Trainer(z, (z+0, z+0), \
            [momentum_sgd(z.parameters, lr_per_sample, momentum_time_constant,
                True)])

    i = 0
    while i<100:
        i+=1
        input_data = np.random.rand(dim)
        trainer.train_minibatch([input_data])

    assert m.forward_calls == m.backward_calls == 100

def test_udf_clone():
    dim = 4
    i = input_variable(dim, needs_gradient=True, name='i_var')
    m_udf = user_function(MyPlus(i, constant(3)))
    p = parameter(shape=(dim,), init=10, name='p')
    z = m_udf + p
    
    z_clone = z.clone('share');

    input_data = np.random.rand(dim)
    result = z_clone.eval([input_data])
    assert np.allclose(result[0][0], input_data+3+10)


@pytest.mark.parametrize("payload", [
    (np.asarray([[[1,2,3.0]]]),),
    (77,),
    ("a", 2),
    (),
    (None)
    ])
def test_ext_backpropstate(payload):

    class TestBackPropState(UserFunction):
        def __init__(self, arg, payload, name='f1'):
            self.payload = payload
            super(TestBackPropState, self).__init__([arg])

        def infer_outputs(self):
            return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes)]

        def forward(self, argument, device=None, outputs_to_retain=None):
            return self.payload, argument

        def backward(self, state, root_gradients):
            assert state == self.payload
            return root_gradients

    dim = 4

    p = parameter(shape=(dim,), init=10)
    in1 = input_variable(dim, needs_gradient=True, name='i_var')
    m = user_function(TestBackPropState(in1, payload))
    z = m+p

    lr_per_sample = learning_rate_schedule(0.007, UnitType.sample)
    trainer = Trainer(z, (z+0, z+0), \
            [sgd(z.parameters, lr_per_sample)])

    for i in range(100):
        input_data = np.random.rand(dim)
        trainer.train_minibatch({in1:[input_data]})

class LambdaFunc(UserFunction):
    def __init__(self,
            arg,
            when=lambda arg: True,
            execute=lambda arg: print(arg),
            name=''):
        self.when = when
        self.execute = execute
        super(LambdaFunc, self).__init__([arg], name=name)

    def infer_outputs(self):
        return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes)]

    def forward(self, argument, device=None, outputs_to_retain=None):
        if self.when(argument):
            self.execute(argument)

        return None, argument

    def backward(self, state, root_gradients):
        return root_gradients


def test_ext_lambdafunc():
    dim = 4

    class CallbackCounter(object):
        def __init__(self):
            self.count = 0
        def inc(self, arg):
            self.count += 1

    cb = CallbackCounter()

    p = parameter(shape=(dim,), init=1)
    i = input_variable(dim, needs_gradient=True, name='i_var')
    k = i*p
    m = LambdaFunc(k,
            when=lambda arg: np.sum(arg)>1,
            execute=cb.inc)
    m = user_function(m)
    z = m+0

    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    lr_per_sample = learning_rate_schedule(0.007, UnitType.sample)
    trainer = Trainer(z, (z+0, z+0), \
            [momentum_sgd(z.parameters, lr_per_sample, momentum_time_constant,
                True)])

    i = 0
    input_data = 0.1 * np.ones(dim)
    trainer.train_minibatch([input_data])
    assert cb.count == 0

    input_data = 0.3 * np.ones(dim)
    trainer.train_minibatch([input_data])
    assert cb.count == 1

class PlusAndLast(UserFunction):
    impl_func = None

    def __init__(self, arg1, arg2, name='f1'):
        i1 = input_variable(arg1.shape, arg1.dtype, name='i1', dynamic_axes=arg1.dynamic_axes)
        i2 = input_variable(arg2.shape, arg2.dtype, name='i2', dynamic_axes=arg2.dynamic_axes)
        self.impl_func = sequence.last(i1 + sequence.broadcast_as(i2, i1))

        super(PlusAndLast, self).__init__([arg1, arg2], name=name)

    def infer_outputs(self):
        impl_func_output = self.impl_func.output
        return [output_variable(impl_func_output.shape, impl_func_output.dtype, impl_func_output.dynamic_axes)]

    def forward(self, arguments, device=None, outputs_to_retain=None):
        _, result = self.impl_func.forward({self.impl_func.arguments[0] : arguments[0], self.impl_func.arguments[1] : arguments[1]}, [self.impl_func.output])
        return None, result[self.impl_func.output]

def test_udf_plus_and_last():
    x = input_variable(shape=(2,))
    y = input_variable(shape=(2,), dynamic_axes=[Axis.default_batch_axis()])
    
    func = user_function(PlusAndLast(x, y))

    dt_precision = np.float32
    operand1 = [AA([[1., 2.], [3., 4.]], dtype=dt_precision)]
    operand2 = [AA([2., 2.], dtype=dt_precision)]

    _, result = func.forward({x : operand1, y : operand2}, [func.output])
    
    expected_forward = AA([[[5., 6.]]], dtype=dt_precision)
    assert np.allclose(result[func.output], expected_forward)
