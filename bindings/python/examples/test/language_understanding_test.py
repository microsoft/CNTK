# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
from cntk import DeviceDescriptor

# this emulates a "main" function for SLUHandsOn
def _run_lu_1():
    from examples.language_understanding.language_understanding import data_dir, create_reader, create_model, train
    from _cntk_py import set_fixed_random_seed
    set_fixed_random_seed(1) # to become invariant to initialization order, which is a valid change
    reader = create_reader(data_dir + "/atis.train.ctf")
    model = create_model()
    loss, metric = train(reader, model, max_epochs=1)
    return metric, loss  # note: strange order

TOLERANCE_ABSOLUTE = 1E-1

def test_seq_classification_error(device_id):
    from cntk.utils import cntk_device
    DeviceDescriptor.set_default_device(cntk_device(device_id))

    evaluation_avg, loss_avg = _run_lu_1()

    # ng] loss = 0.783951 * 36061, metric = 15.5% * 3606
    expected_avg = [0.15570838301766451, 0.7846451368305728]
    assert np.allclose([evaluation_avg, loss_avg], expected_avg, atol=TOLERANCE_ABSOLUTE)

if __name__=='__main__':
    test_seq_classification_error(0)
