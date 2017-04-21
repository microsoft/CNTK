# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import sys
import pytest

_DEFAULT_DEVICE_ID = -1


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

        metafunc.parametrize("device_id", devices)

    if 'is_1bit_sgd' in metafunc.fixturenames:
        if (len(metafunc.config.option.is1bitsgd)) > 1:
            del metafunc.config.option.is1bitsgd[0]

        is1bitsgd = set()
        for elem in metafunc.config.option.is1bitsgd:
            if elem == "0" or elem == "1":
                is1bitsgd.add(int(elem))
            else:
                raise RuntimeError("invalid is1bitsgd value {}, only 0 or 1 allowed".format(elem))

        metafunc.parametrize("is_1bit_sgd", is1bitsgd)

#
# Adding the namespaces so that doctests work
#
import numpy
# Because of difference in precision across platforms, we restrict the output
# precision and don't write in scientific notation
numpy.set_printoptions(precision=6, suppress=True)

import cntk.debugging
cntk.debugging.set_computation_network_track_gap_nans(True)

import cntk
@pytest.fixture(autouse=True)
def add_namespace(doctest_namespace):
    doctest_namespace['np'] = numpy
    doctest_namespace['C'] = cntk
