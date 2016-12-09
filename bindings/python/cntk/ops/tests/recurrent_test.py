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
        unittest_helper, I, sanitize_dtype_cntk
from .. import parameter

SEQUENCES = [
    # (shape of batch (sample size, seq size, rows, cols), time step, initial state)
    # ((1, 4, 3, 2), 1, 0.1),
    ((2, 2, 4, 2), 1, 0.5),
    # ((2, 2, 4, 2), 2, 0.3)
]

@pytest.mark.parametrize("input_size, time_step, initial_state", SEQUENCES)
def test_op_future_value(input_size, time_step, initial_state, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    total_elements = np.product(input_size)

    x = np.arange(total_elements, dtype=dt).reshape(input_size)
    elements_to_roll = total_elements - np.product(input_size[2:]) * time_step
    x_rolled = np.roll(AA(x, dtype=dt), elements_to_roll)
    np.put(x_rolled, range(elements_to_roll, total_elements), initial_state)

    expected_forward = AA([x_rolled], dtype=dt)

    backward = np.ones_like(x_rolled, dtype=dt)
    np.put(backward, range(total_elements - elements_to_roll), 0.0)

    a = I(shape=x.shape[2:],
      dtype=sanitize_dtype_cntk(precision),
      needs_gradient=True,
      name='a')

    expected_backward = {
        a: [backward]
    }

    init = parameter(init=AA(initial_state, dtype=dt))

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
    x = np.arange(total_elements, dtype=dt).reshape(input_size)

    for seq_idx in range(input_size[0]):
        import ipdb;ipdb.set_trace()
        x[seq_idx] = np.roll(AA(x[seq_idx], dtype=dt), time_step, axis=0)
        x[seq_idx,0:time_step] = initial_state

    expected_forward = x

    a = I(shape=x.shape[2:],
      dtype=sanitize_dtype_cntk(precision),
      needs_gradient=True,
      name='a')

    backward = np.ones_like(x, dtype=dt)
    np.put(backward, range(total_elements - elements_to_roll, total_elements), 0.0)

    expected_backward = {
        a: backward
    }

    init = parameter(init=AA(initial_state, dtype=dt))

    from .. import past_value
    input_op_input = past_value(a, init, time_step)

    unittest_helper(input_op_input,
                x, expected_forward, expected_backward,
                device_id=device_id, precision=precision)
