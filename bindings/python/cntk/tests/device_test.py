# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for the device module
"""

from ..device import DeviceKind, cpu, gpu, all_devices
from .. import cntk_py


def test_device_kind():
    assert cpu().type() == DeviceKind.CPU
    assert cpu().type() != DeviceKind.GPU

    for d in all_devices():
        if d.type() != cntk_py.DeviceKind_GPU:
            continue
        assert d.type() == DeviceKind.GPU
        assert d.type() != DeviceKind.CPU
