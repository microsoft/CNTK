# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import math
import numpy as np
from .. import Function
from ..trainer import *
from ..learner import *
from .. import cross_entropy_with_softmax, classification_error, parameter, \
        input_variable, times, plus, reduce_sum

def test_trainer(tmpdir):
    in1 = input_variable(shape=(1,))
    labels = input_variable(shape=(1,))
    p = parameter(shape=(2,), init=10)
    z = plus(in1, reduce_sum(p), name='z')
    ce = cross_entropy_with_softmax(z, labels)
    errs = classification_error(z, labels)

    m_schedule = momentum_schedule(1100)

    trainer = Trainer(z, ce, errs, \
            [sgd(z.parameters, 0.007, m_schedule, 0.5, True)])
    in1_value = [[1],[2]]
    label_value = [[0], [1]]
    arguments = {in1: in1_value, labels: label_value}
    z_output = z.output
    updated, var_map = trainer.train_minibatch(arguments, [z_output])

    p = str(tmpdir / 'checkpoint.dat')
    trainer.save_checkpoint(p)
    trainer.restore_from_checkpoint(p)

    assert trainer.model.name == 'z'

    # Ensure that Swig is not leaking raw types
    assert isinstance(trainer.model, Function)
    assert trainer.model.__doc__
    assert isinstance(trainer.parameter_learners[0], Learner)

def test_output_to_retain():
    in1 = input_variable(shape=(1,))
    labels = input_variable(shape=(1,))
    p = parameter(shape=(2,), init=10)
    z = plus(in1, reduce_sum(p), name='z')
    ce = cross_entropy_with_softmax(z, labels)
    errs = classification_error(z, labels)

    m_schedule = momentum_schedule(1100)

    trainer = Trainer(z, ce, errs, \
            [sgd(z.parameters, 0.007, m_schedule, 0.5, True)])
    in1_value = [[1],[2]]
    label_value = [[0], [1]]
    arguments = {in1: in1_value, labels: label_value}
    z_output = z.output
    updated, var_map = trainer.train_minibatch(arguments, [z_output])

    assert np.allclose(var_map[z_output], np.asarray(in1_value)+20)


