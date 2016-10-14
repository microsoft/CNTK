# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# TODO: This does not work yet, need to figure out the right pattern.

import numpy as np
from cntk import DeviceDescriptor

# this emulates a "main" function for SLUHandsOn
def slu_hands_on():
    import examples.SLUHandsOn.SLUHandsOn  # this runs the entire thing
    # No, not working. Won't return anything, different scope.
    return metric, loss  # note: strange order

TOLERANCE_ABSOLUTE = 1E-1

def test_seq_classification_error(device_id):
    from cntk.utils import cntk_device
    DeviceDescriptor.set_default_device(cntk_device(device_id))

    evaluation_avg, loss_avg = slu_hands_on()

    expected_avg = [0.55, 1.53099]
    assert np.allclose([evaluation_avg, loss_avg], expected_avg, atol=TOLERANCE_ABSOLUTE)

#test_seq_classification_error(0)  # uncomment this to run the test explicitly
