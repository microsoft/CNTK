# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
from cntk import DeviceDescriptor

TOLERANCE_ABSOLUTE = 1E-1

from cntk.blocks import *
from cntk.layers import *
from cntk.models import *
from cntk.utils import *
from examples.LanguageUnderstanding.LanguageUnderstanding import data_dir, create_reader, create_model, train, emb_dim, hidden_dim, label_dim

def create_test_model():
    # this selects additional nodes and alternative paths
    with default_options(enable_self_stabilization=True, use_peepholes=True):
        return Sequential([
            Stabilizer(),
            Embedding(emb_dim),
            BatchNormalization(),
            Recurrence(LSTM(hidden_dim, cell_shape=hidden_dim+50), go_backwards=True),
            BatchNormalization(map_rank=1),
            Dense(label_dim)
        ])

def test_seq_classification_error(device_id):
    from cntk.utils import cntk_device
    DeviceDescriptor.set_default_device(cntk_device(device_id))

    from _cntk_py import set_computation_network_trace_level, set_fixed_random_seed
    set_computation_network_trace_level(1)
    set_fixed_random_seed(1) # to become invariant to initialization order, which is a valid change

    # test of the example itself
    # this emulates the main code in the PY file
    reader = create_reader(data_dir + "/atis.train.ctf")
    model = create_model()
    loss_avg, evaluation_avg = train(reader, model, max_epochs=1)
    expected_avg = [0.15570838301766451, 0.7846451368305728]
    assert np.allclose([evaluation_avg, loss_avg], expected_avg, atol=TOLERANCE_ABSOLUTE)

    # test of a config like in the example but with additions to test many code paths
    if device_id >= 0: # BatchNormalization currently does not run on CPU
        reader = create_reader(data_dir + "/atis.train.ctf")
        model = create_test_model()
        loss_avg, evaluation_avg = train(reader, model, max_epochs=1)
        log_number_of_parameters(model, trace_level=1) ; print()
        expected_avg = [0.084, 0.407364]
        assert np.allclose([evaluation_avg, loss_avg], expected_avg, atol=TOLERANCE_ABSOLUTE)

if __name__=='__main__':
    test_seq_classification_error(0)
