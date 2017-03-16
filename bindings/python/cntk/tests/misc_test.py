# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import pytest

import cntk

def test_callstack1():
    with pytest.raises(ValueError) as excinfo:
        cntk.device.gpu(99999)
    assert '[CALL STACK]' in str(excinfo.value)

def test_callstack2():
    with pytest.raises(ValueError) as excinfo:
        cntk.io.MinibatchSource(cntk.io.CTFDeserializer("", streams={}))
    assert '[CALL STACK]' in str(excinfo.value)

def test_Value_raises():
    from cntk import NDArrayView, Value
    with pytest.raises(ValueError):
        nd = NDArrayView.from_dense(np.asarray([[[4,5]]], dtype=np.float32))
        val = Value(nd)
