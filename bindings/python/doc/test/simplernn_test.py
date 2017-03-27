# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import sys
import numpy as np

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, ".."))
from simplernn import train_sequence_classifier

TOLERANCE_ABSOLUTE = 5E-2


def test_rnn_error(device_id):
    error, loss = train_sequence_classifier()

    expected_error = 0.333333
    expected_loss  = 1.12

    assert np.allclose(error, expected_error, atol=TOLERANCE_ABSOLUTE)
    assert np.allclose(loss, expected_loss, atol=TOLERANCE_ABSOLUTE)
