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
    parser.addoption("--is1bitsgd", default="0",
                     help="whether 1-bit SGD is used")

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

    if 'is_1bit_sgd' in metafunc.fixturenames:
        if (len(metafunc.config.option.is1bitsgd)) > 1:
            del metafunc.config.option.is1bitsgd[0]

        is1bitsgd = set()
        for elem in metafunc.config.option.is1bitsgd:
            if elem == "0" or elem == "1":
                is1bitsgd.add(int(elem))
            else:
                raise RuntimeError("invalid is1bitsgd value {}, only 0 or 1 allowed".format(elem))

        metafunc.parametrize("is_1bit_sgd", is1bitsgd, scope='session')

@pytest.fixture(autouse=True)
def reset_random_seed():
    cntk.cntk_py.reset_random_seed(0)
