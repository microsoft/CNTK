# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import sys
import pytest

_DEFAULT_DEVICE_ID=-1

import cntk.cntk_py
cntk.cntk_py.always_allow_setting_default_device()

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

@pytest.fixture(scope='module')
def nb(tmpdir_factory, request, device_id):
    import nbformat
    import os
    import subprocess
    from cntk.ops.tests.ops_test_utils import cntk_device
    from cntk.cntk_py import DeviceKind_GPU
    inPath = getattr(request.module, "notebook")

    deviceIdsToRun = [-1, 0]
    try:
        deviceIdsToRun = getattr(request.module, "notebook_deviceIdsToRun")
    except AttributeError:
        pass

    timeoutSeconds = 450
    try:
        timeoutSeconds = int(getattr(request.module, "notebook_timeoutSeconds"))
    except AttributeError:
        pass

    # Pass along device_id type to child process
    if cntk_device(device_id).type() == DeviceKind_GPU:
        os.environ['TEST_DEVICE'] = 'gpu'
    else:
        os.environ['TEST_DEVICE'] = 'cpu'
    if not device_id in deviceIdsToRun:
        pytest.skip('test not configured to run on device ID {0}'.format(device_id))
    outPath = str(tmpdir_factory.mktemp('notebook').join('out.ipynb'))
    assert os.path.isfile(inPath)
    kernel_name_opt = "--ExecutePreprocessor.kernel_name=python%d" % (sys.version_info[0])
    args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
            "--ExecutePreprocessor.timeout={0}".format(timeoutSeconds),
            kernel_name_opt, "--output", outPath, inPath]
    subprocess.check_call(args)
    nb = nbformat.read(outPath, nbformat.current_nbformat)
    return nb
