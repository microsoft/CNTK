# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# TODO: This does not work yet, need to figure out the right pattern.

import numpy as np
from cntk import DeviceDescriptor

# this emulates a "main" function for SLUHandsOn
from examples.SLUHandsOn.SLUHandsOn import *
from examples.SLUHandsOn.SLUHandsOn import _Infer  # TODO: remove
def slu_hands_on():
    reader = create_reader(data_dir + "/atis.train.ctf")
    model = create_model(_inf=_Infer(shape=input_dim, axis=[Axis.default_batch_axis(), Axis.default_dynamic_axis()]))
    loss, metric = train(reader, model, max_epochs=1)
    return metric, loss  # note: strange order

TOLERANCE_ABSOLUTE = 1E-1

def test_seq_classification_error(device_id):
    from cntk.utils import cntk_device
    DeviceDescriptor.set_default_device(cntk_device(device_id))

    evaluation_avg, loss_avg = slu_hands_on()

    expected_avg = [0.15570838301766451, 0.7846451368305728]
    assert np.allclose([evaluation_avg, loss_avg], expected_avg, atol=TOLERANCE_ABSOLUTE)

if __name__=='__main__':
    test_seq_classification_error(0)
