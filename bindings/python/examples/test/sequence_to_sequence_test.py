# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
from cntk.device import set_default_device

from examples.Sequence2Sequence.Sequence2Sequence import sequence_to_sequence_translator

TOLERANCE_ABSOLUTE = 1E-1

def test_sequence_to_sequence(device_id):
    from cntk.utils import cntk_device
    set_default_device(cntk_device(device_id))

    error = sequence_to_sequence_translator(False, True)

    expected_error =  0.830361
    assert np.allclose(error, expected_error, atol=TOLERANCE_ABSOLUTE)
