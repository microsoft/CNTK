# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import cntk as C


def test_attention_model():
    attention_dim = 128
    attention_span = 20
    attention_axis = -3

    att_model = C.layers.AttentionModel(attention_dim, attention_span, attention_axis, name='attention_model')

    expected_num_of_inputs = 142

    assert len(att_model.inputs) == expected_num_of_inputs
