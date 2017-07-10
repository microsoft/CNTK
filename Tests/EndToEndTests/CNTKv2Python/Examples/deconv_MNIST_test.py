# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os, sys
import numpy as np
import subprocess
from cntk.device import try_set_default_device, gpu

abs_path = os.path.dirname(os.path.abspath(__file__))
getting_started_path = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "GettingStarted")
sys.path.append(getting_started_path)
autoEncoder = __import__("07_Deconvolution_PY")
visualizer = __import__("07_Deconvolution_Visualizer")

TOLERANCE_ABSOLUTE = 5E-1

def test_simple_mnist_py_error(device_id):
    from cntk.ops.tests.ops_test_utils import cntk_device
    try_set_default_device(cntk_device(device_id))

    test_rmse = autoEncoder.deconv_mnist(max_epochs=1)
    expected_test_rmse = 0.288

    assert np.allclose(test_rmse, expected_test_rmse,
                       atol=TOLERANCE_ABSOLUTE)

    # check that it doesn't break
    visualizer.generate_visualization(use_brain_script_model=False, testing=True)

def test_simple_mnist_bs_error(device_id):
    from cntk.ops.tests.ops_test_utils import cntk_device
    try_set_default_device(cntk_device(device_id))

    cntkPath = os.environ["TEST_CNTK_BINARY"]
    if sys.platform == "win32":
        p = subprocess.Popen(["cygpath", "-aw", os.environ["TEST_CNTK_BINARY"]], stdout=subprocess.PIPE)
        out = p.communicate()[0]
        cntkPath = out.decode(sys.getdefaultencoding()).strip()

    pid = subprocess.Popen([cntkPath, "configFile=07_Deconvolution_BS.cntk"], cwd=getting_started_path)
    pid.wait()

    assert pid.returncode == 0, "ERROR: cntk ended with exit code {}".format(pid.returncode)

    # check that it doesn't break
    visualizer.generate_visualization(use_brain_script_model=True, testing=True)
