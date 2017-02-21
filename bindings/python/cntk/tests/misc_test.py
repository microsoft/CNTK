# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import cntk
import pytest

def test_callstack1():
    with pytest.raises(ValueError) as excinfo:
        cntk.device.gpu(99999)
    assert '[CALL STACK]' in str(excinfo.value)

def test_callstack2():
    with pytest.raises(ValueError) as excinfo:
        cntk.io.MinibatchSource(cntk.io.CTFDeserializer("", streams={}))
    assert '[CALL STACK]' in str(excinfo.value)
