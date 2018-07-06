#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import onnx.backend.test

from onnx_cntk import backend as cntk_backend

# This is a pytest magic variable to load extra plugins
pytest_plugins = 'onnx.backend.test.report'

backend_test = onnx.backend.test.BackendTest(cntk_backend, __name__)

skip_models = [
    'bvlc_alexnet',
    'densenet121',
    'inception_v1',
    'inception_v2',
    'resnet50',
    'shufflenet',
    'vgg16',
    'vgg19',
]

skip_ops = [
    'test_max',
    'test_min',
]

skip_tests = skip_models + skip_ops
for test in skip_tests:
    backend_test.exclude(test)

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test
                 .enable_report()
                 .test_cases)

if __name__ == '__main__':
    unittest.main()