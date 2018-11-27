# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import sys
import pytest

_DEFAULT_DEVICE_ID=-1

import cntk
import cntk.debugging
cntk.cntk_py.always_allow_setting_default_device()
cntk.debugging.set_checked_mode(True)

def pytest_addoption(parser):
    parser.addoption("--deviceid", action="append", default=[_DEFAULT_DEVICE_ID],
        help="list of device ids to pass to test functions")

DEVICE_MAP = {
        'auto': 'auto',
        'cpu': -1,
        'gpu': 0
        }

def pytest_generate_tests(metafunc):
    if 'device_id' in metafunc.fixturenames:
        if (len(metafunc.config.option.deviceid)) > 1:
            del metafunc.config.option.deviceid[0]

        devices = set()
        for elem in metafunc.config.option.deviceid:
            try:
                if elem in DEVICE_MAP:
                    devices.add(DEVICE_MAP[elem])
                else:
                    devices.add(int(elem))
            except ValueError:
                raise RuntimeError("invalid deviceid value '{0}', please " +
                    "use integer values or 'auto'".format(elem))

        metafunc.parametrize("device_id", devices, scope='session')

@pytest.fixture(autouse=True)
def reset_random_seed():
    cntk.cntk_py.reset_random_seed(0)
