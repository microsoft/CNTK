# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for recurrent operations, each operation is tested for
the forward and the backward pass
"""

from __future__ import division
import numpy as np
import pytest
from .ops_test_utils import _test_binary_op, AA, precision, PRECISION_TO_TYPE,\
        unittest_helper, I, sanitize_dtype_cntk, cntk_device
from .. import parameter

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

    a = I(shape=elem_shape,
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

    elem_shape = input_size[2:]

    elements_to_roll = np.product(elem_shape) * time_step
    x = np.arange(total_elements, dtype=dt).reshape(input_size)

    expected_forward = np.zeros_like(x, dtype=dt)

    for seq_idx in range(input_size[0]):
        expected_forward[seq_idx] = np.roll(AA(x[seq_idx], dtype=dt), time_step, axis=0)
        expected_forward[seq_idx,0:time_step] = initial_state

    a = I(shape=elem_shape,
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
