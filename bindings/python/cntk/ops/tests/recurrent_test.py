# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for recurrent operations, each operation is tested for
the forward and the backward pass
"""

# BUGBUG: These are not testing actual execution in a loop.

from __future__ import division
import numpy as np
import pytest
from .ops_test_utils import _test_binary_op, AA, precision, PRECISION_TO_TYPE,\
        unittest_helper, sanitize_dtype_cntk, cntk_device
from .. import parameter, sequence

SEQUENCES = [
    # (shape of batch (sample size, seq size, rows, cols), time step, initial state)
    ((1, 4, 3, 2), 1, 0.1),
    ((2, 2, 4, 2), 1, 0.5),
    ((2, 2, 4, 2), 2, 0.3)
]

@pytest.mark.parametrize("input_size, time_step, initial_state", SEQUENCES)
def test_op_future_value(input_size, time_step, initial_state, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    expected_forward_list = []
    expected_backward_list = []

    num_of_seq = input_size[0]
    total_elements_in_seq = np.product(input_size[1:])
    x = np.arange(total_elements_in_seq * num_of_seq, dtype=dt).reshape(input_size)
    elem_shape = input_size[2:]
    for seq in x:
        elements_to_roll = total_elements_in_seq - np.product(elem_shape) * time_step
        x_rolled = np.roll(AA(seq, dtype=dt), elements_to_roll)
        np.put(x_rolled, range(elements_to_roll, total_elements_in_seq), initial_state)

        expected_forward_list.append(AA(x_rolled, dtype=dt))
        backward = np.ones_like(x_rolled, dtype=dt)
        np.put(backward, range(total_elements_in_seq - elements_to_roll), 0.0)
        expected_backward_list.append(backward)

    expected_forward = AA(expected_forward_list, dtype=dt)

    a = sequence.input(shape=elem_shape,
                       dtype=sanitize_dtype_cntk(precision),
                       needs_gradient=True,
                       name='a')

    expected_backward = {
        a: AA(expected_backward_list, dtype=dt)
    }
    init = parameter(init=AA(initial_state, dtype=dt), device=cntk_device(device_id))

    from .. import future_value
    input_op_input = future_value(a, init, time_step)

    unittest_helper(input_op_input,
                x, expected_forward, expected_backward,
                device_id=device_id, precision=precision)

@pytest.mark.parametrize("input_size, time_step, initial_state", SEQUENCES)
def test_op_past_value(input_size, time_step, initial_state, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    total_elements = np.product(input_size)

    x = np.arange(total_elements, dtype=dt).reshape(input_size)

    expected_forward = np.zeros_like(x, dtype=dt)

    for seq_idx in range(input_size[0]):
        expected_forward[seq_idx] = np.roll(AA(x[seq_idx], dtype=dt), time_step, axis=0)
        expected_forward[seq_idx,0:time_step] = initial_state

    elem_shape = input_size[2:]
    a = sequence.input(shape=elem_shape,
                       dtype=sanitize_dtype_cntk(precision),
                       needs_gradient=True,
                       name='a')

    backward = np.ones_like(x, dtype=dt)
    for seq_idx in range(input_size[0]):
        for t in range(time_step):
            backward[seq_idx,-1-t] = 0.0

    expected_backward = {
        a: backward
    }

    init = parameter(init=AA(initial_state, dtype=dt), device=cntk_device(device_id))

    from .. import past_value
    input_op_input = past_value(a, init, time_step)

    unittest_helper(input_op_input,
                x, expected_forward, expected_backward,
                device_id=device_id, precision=precision)

SEQUENCES1 = [
    # (shape of batch (sample num, seq len, rows, cols), time step, initial state size)
    ((1, 4, 3, 2),  1, (1, 1, 3, 2)),  # 1 step, initial state has len 1
    ((1, 4, 3, 2),  1, (1, 3, 3, 2)),  # 1 step, initial state has len 3, must pick out last
    ((1, 4, 3, 2), -1, (1, 1, 3, 2)),  # 1 step, initial state has len 1
    ((1, 4, 3, 2), -1, (1, 2, 3, 2)),  # 1 step, initial state has len 2, must pick out first
    ((2, 4, 4, 1), -2, (2, 1, 4, 1)),  # 2 seqs of 4 steps each, initial state will broadcast temporally  --note: broadcasting shape-wise at the same time is not supported presently by gather()
    ((2, 4, 4, 3),  2, (2, 3, 4, 3))   # 2 seqs of 4 steps each, initial state has different length and will not broadcast
    # note: not tested yet: if time_step > length of initial_state (and initial_state > 1 frame i.e. not broadcasting), it should throw
]

@pytest.mark.parametrize("input_size, time_step, initial_state_size", SEQUENCES1)
def test_op_delay_with_initial_state(input_size, time_step, initial_state_size, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    x             = np.arange(np.product(input_size),         dtype=dt).reshape(input_size)
    initial_state = np.arange(np.product(initial_state_size), dtype=dt).reshape(initial_state_size) + 3

    expected_forward = np.zeros_like(x, dtype=dt)
    for seq_idx in range(input_size[0]):
        expected_forward[seq_idx] = np.roll(AA(x[seq_idx], dtype=dt), time_step, axis=0)
        if time_step > 0:
            expected_forward[seq_idx,:time_step] = initial_state[seq_idx,-1 if time_step == 1 else -time_step:,...]
        else:
            expected_forward[seq_idx,time_step:] = initial_state[seq_idx,:1 if time_step == -1 else -time_step,...]

    a = sequence.input(shape=input_size[2:],
                       dtype=sanitize_dtype_cntk(precision),
                       needs_gradient=True,
                       name='a')
    from ...axis import Axis
    i = sequence.input(shape=initial_state_size[2:],
                       dtype=sanitize_dtype_cntk(precision),
                       needs_gradient=True,
                       sequence_axis=Axis('initial_state_axis'),
                       name='i')

    backward = np.ones_like(x, dtype=dt)
    initial_state_backward = np.zeros_like(initial_state, dtype=dt)
    for seq_idx in range(input_size[0]):
        for t in range(abs(time_step)):
            initial_state_bw_val = 1.0 if initial_state_size[1] > 1 else abs(time_step)
            if time_step > 0:
                backward[seq_idx,-1-t] = 0.0
                if t < initial_state_size[1]:
                    initial_state_backward[seq_idx,-1-t] = initial_state_bw_val
            else:
                backward[seq_idx,t] = 0.0
                if t < initial_state_size[1]:
                    initial_state_backward[seq_idx,t] = initial_state_bw_val

    print(initial_state_size)
    print(initial_state_backward)

    expected_backward = {
        a: backward,
        i: initial_state_backward
    }

    input_op_input = sequence.delay(a, i, time_step)

    unittest_helper(input_op_input,
                {a: x, i: initial_state}, expected_forward, expected_backward,
                device_id=device_id, precision=precision)
