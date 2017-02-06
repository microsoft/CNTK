# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os, sys
import numpy as np
from cntk.device import set_default_device

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Text", "CharacterLM"))
from char_rnn import train_and_eval_char_rnn

def test_char_rnn(device_id):
    from cntk.ops.tests.ops_test_utils import cntk_device
    set_default_device(cntk_device(device_id))

    # Just run and verify it does not crash
    output = train_and_eval_char_rnn(200)
    print(output)
