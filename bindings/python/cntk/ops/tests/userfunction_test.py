# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for function extension
"""

from __future__ import division
import numpy as np
import pytest

from cntk import *
from cntk.trainer import *
from cntk.learner import *
from cntk.ops.functions import UserFunction

class Plus3Func(UserFunction):
    def __init__(self, arg, name='f1'):
        outputs = [output_variable(arg.shape, arg.dtype, arg.dynamic_axes)]
        super(Plus3Func, self).__init__([arg], outputs,
                name=name)

    def forward(self, arguments, outputs, device=None, outputs_to_retain=None):
        assert len(self.inputs)==1
        assert len(arguments)==1
        assert len(outputs)==1

        for k in outputs:
            outputs[k] = arguments[0] + 3
            break

        return None, outputs

    def backward(self, state, root_gradients, variables):
        assert len(root_gradients) == 1
        assert len(variables) == 1

        for rk, rv in root_gradients.items():
            break
        for var_key in variables:
            break

        variables[var_key] = rv

def test_ext_eval_1():
    dim = 4
    p = parameter(shape=(dim,), init=10, name='p')
    i = input_variable(dim, needs_gradient=True, name='i_var')
    m = Plus3Func(i)
    z = m+p

    input_data = np.random.rand(dim)
    result = z.eval([input_data])
    assert np.allclose(result[0][0], input_data+3+10)

def test_ext_eval_2_only_param():
    dim = 4
    p = parameter(shape=(dim,), init=10, name='p')
    i = input_variable(dim, needs_gradient=True, name='i_var')
    m = Plus3Func(p)
    # combine does not work
    # z = combine([m.output])
    z = m+i

    input_data = np.random.rand(dim)
    result = z.eval([input_data])
    assert np.allclose(result[0][0], input_data+3+10)

def test_ext_eval_3_no_input():
    dim = 4
    p = parameter(shape=(dim,), init=10, name='p')
    m = Plus3Func(p)
    z = m+0

    result = z.eval()
    # No batch dimension since we have no input
    assert np.allclose(result, np.zeros_like(p)+10+3)

def test_ext_eval_4_a_inside_graph():
    dim = 4
    p_init = 10
    p = parameter(shape=(dim,), init=p_init, name='p')
    m = Plus3Func(p)
    z = p * m

    result = z.eval()
    # No batch dimension since we have no input
    assert np.allclose(result, ((p_init*np.ones_like(result))+3)*p_init)

def _test_ext_eval_4_b_inside_graph():
    dim = 4
    p_init = 10
    p = parameter(shape=(dim,), init=p_init, name='p')
    z = p * Plus3Func(p)

    result = z.eval()
    # No batch dimension since we have no input
    assert np.allclose(result, ((p_init*np.ones_like(result))+3)*p_init)


# TODO change to real training example
def test_ext_train():
    dim = 4

    p = parameter(shape=(dim,), init=10)
    i = input_variable(dim, needs_gradient=True, name='i_var')
    m = Plus3Func(i)
    z = m+p

    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    lr_per_sample = learning_rate_schedule(0.007, UnitType.sample)
    trainer = Trainer(z, z+0, z+0, \
            [momentum_sgd(z.parameters, lr_per_sample, momentum_time_constant,
                True)])

    i = 0
    while i<100:
        i+=1
        input_data = np.random.rand(dim)
        trainer.train_minibatch([input_data])

@pytest.mark.parametrize("payload", [
    (77,),
    ("a", 2),
    (),
    (None)
    ])
def test_ext_backpropstate(payload):

    class TestBackPropState(UserFunction):
        def __init__(self, arg, payload, name='f1'):
            outputs = [output_variable(arg.shape, arg.dtype, arg.dynamic_axes)]
            self.payload = payload
            super(TestBackPropState, self).__init__([arg], outputs)

        def forward(self, arguments, outputs, device=None, outputs_to_retain=None):
            for k in outputs:
                outputs[k] = arguments[0]
                break

            return self.payload, outputs

        def backward(self, state, root_gradients, variables):
            assert state == self.payload
            for rk, rv in root_gradients.items():
                break
            for var_key in variables:
                break

            variables[var_key] = rv

    dim = 4

    p = parameter(shape=(dim,), init=10)
    i = input_variable(dim, needs_gradient=True, name='i_var')
    m = TestBackPropState(i, payload)
    z = m+p

    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    lr_per_sample = learning_rate_schedule(0.007, UnitType.sample)
    trainer = Trainer(z, z+0, z+0, \
            [momentum_sgd(z.parameters, lr_per_sample, momentum_time_constant,
                True)])

    i = 0
    input_data = np.random.rand(dim)
    trainer.train_minibatch([input_data])

class LambdaFunc(UserFunction):
    def __init__(self,
            arg,
            when=lambda arg: True,
            execute=lambda arg:self.print_tensor(arg),
            name=''):
        self.when = when
        self.execute = execute
        outputs = [output_variable(arg.shape, arg.dtype, arg.dynamic_axes)]
        super(LambdaFunc, self).__init__([arg], outputs,
                name=name)

    def forward(self, arguments, outputs, device=None, outputs_to_retain=None):
        if len(arguments)!=1:
            raise ValueError('LambdaFunc expects exactly one input')

        if self.when(arguments):
            self.execute(arguments)

        for k in outputs:
            outputs[k] = arguments[0]
            break

        return None, outputs

    def backward(self, state, root_gradients, variables):
        for rk, rv in root_gradients.items():
            break
        for var_key in variables:
            break

        variables[var_key] = rv


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
    z = m+0

    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    lr_per_sample = learning_rate_schedule(0.007, UnitType.sample)
    trainer = Trainer(z, z+0, z+0, \
            [momentum_sgd(z.parameters, lr_per_sample, momentum_time_constant,
                True)])

    i = 0
    input_data = 0.1 * np.ones(dim)
    trainer.train_minibatch([input_data])
    assert cb.count == 0

    input_data = 0.3 * np.ones(dim)
    trainer.train_minibatch([input_data])
    assert cb.count == 1
