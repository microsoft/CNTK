# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
from cntk import DeviceDescriptor

from examples.SequenceClassification.SequenceClassification import train_sequence_classifier

TOLERANCE_ABSOLUTE = 1E-1

def test_seq_classification_error(device_id):
    from cntk.utils import cntk_device
    DeviceDescriptor.set_default_device(cntk_device(device_id))

    evaluation_avg, loss_avg = train_sequence_classifier()

    expected_avg = [0.55, 1.53099]
    assert np.allclose([evaluation_avg, loss_avg], expected_avg, atol=TOLERANCE_ABSOLUTE)
