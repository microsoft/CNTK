# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import pytest
import numpy as np
import cntk as C

def test_outputs():
    fwd_state = C.placeholder("placeholder")
    prev_state = C.sequence.past_value(fwd_state, name="prev_state")
    z = C.abs(prev_state, "abs")
    output = z.output
    z = z.replace_placeholders({fwd_state: z.output})

    fwd_state = None
    prev_state = None
    z = None

    for arg in output.owner.arguments:
        print("Argument name: {}, argument owner name {}".format(arg.name, arg.owner.name))

def test_0d_data_1d_sample_shape():
    x = C.input_variable(shape=(1,))
    op = x + x

    with pytest.raises(ValueError):
        op.eval({x : [np.asarray(2)]})

def test_1d_NDArrayView_copy():
    x = C.input_variable(shape=(1,))
    op = x + 1
    result = op.eval({x : [np.asarray([1])]}, as_numpy=False)
    result_slice = result.data.slice_view((0, 0), (1,))

    w = C.parameter(init=np.asarray([1]))
    w.set_value(result_slice)

    assert np.array_equal(w.value, result_slice.asarray())

def test_sequences_packed_in_single_ndarray():
    dim = 2
    input_with_sequence_axis = C.sequence.input_variable(shape=(dim,))

    data = np.asarray([[1, 2], [2, 3]])
    op = C.sequence.last(input_with_sequence_axis)
    result = op.eval({input_with_sequence_axis : data})
    assert np.array_equal(result, [[2., 3.]])

    result = op.eval({input_with_sequence_axis : (data, [True, True])})
    assert np.array_equal(result, [[1., 2.], [2., 3.]])


def test_clone_freeze():
    inputs = 3
    outputs = 5

    features = C.input_variable((inputs), np.float32)
    label = C.input_variable((outputs), np.float32)
    weights = C.parameter((inputs, outputs))
    const_weights = C.constant(weights.value)
    z = C.times(features, weights)
    c = C.times(features, const_weights)
    z_clone = z.clone('freeze')
    c_clone = c.clone('freeze')

    # check that z and z_clone are the same
    for p, q in zip(z.parameters, z_clone.constants):
        assert np.array_equal(p.value, q.value)

    # check that c and c_clone are the same
    for p, q in zip(c.constants, c_clone.constants):
        assert np.array_equal(p.value, q.value)

    # keep copies of the old values
    z_copies = [q.value for q in z_clone.constants]
    c_copies = [q.value for q in c_clone.constants]

    # update z
    trainer = C.Trainer(z, C.squared_error(z, label),  C.sgd(z.parameters, C.learning_parameter_schedule(1.0)))
    x = np.random.randn(16,3).astype('f')
    y = np.random.randn(16,5).astype('f')
    trainer.train_minibatch({features: x, label: y})
    # update c
    for cc in c.constants:
        cc.value = np.random.randn(*cc.value.shape).astype('f')

    # check that z changed
    for p, q in zip(z.parameters, z_clone.constants):
        assert not np.array_equal(p.value, q.value)

    # check that z_clone did not change
    for p, q in zip(z_copies, z_clone.constants):
        assert np.array_equal(p, q.value)

    # check that c changed
    for p, q in zip(c.constants, c_clone.constants):
        assert not np.array_equal(p.value, q.value)

    # check that c_clone did not change
    for p, q in zip(c_copies, c_clone.constants):
        assert np.array_equal(p, q.value)
