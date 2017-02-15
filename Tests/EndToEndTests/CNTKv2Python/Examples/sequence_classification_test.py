# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os, sys
import numpy as np
from cntk.device import try_set_default_device

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "SequenceClassification", "SimpleExample", "Python"))
from SequenceClassification import train_sequence_classifier

TOLERANCE_ABSOLUTE = 1E-1

def test_seq_classification_error(device_id):
    from cntk.ops.tests.ops_test_utils import cntk_device
    try_set_default_device(cntk_device(device_id))

    evaluation_avg, loss_avg = train_sequence_classifier()

    expected_avg = [0.51, 1.28]
    assert np.allclose([evaluation_avg, loss_avg], expected_avg, atol=TOLERANCE_ABSOLUTE)
