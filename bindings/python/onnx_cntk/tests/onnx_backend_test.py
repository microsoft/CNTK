# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

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
    # These tests crash and prevent tests after them from running.
]

skip_ops = [
    # Note after the fix for typenameToTypeProto initialization, no tests are crashing anymore.
    # Thus the following is commented out for now, and subject to be removed in the future. 

    # These tests crash and prevent tests after them from running.
    #'test_operator_repeat_dim_overflow_cpu',
    #'test_operator_repeat_dim_overflow_cuda',

    # These use pre-ONNX 1.2 incompatible versions of the operators.

    # This test is incorrect in ONNX 1.2.
    # https://github.com/onnx/onnx/issues/1210
    #'test_log_softmax_lastdim_cpu',
    #'test_log_softmax_lastdim_cuda',

    # This is an experimental op.
    #'test_thresholdedrelu',

    # Unfinished local test implementation.
    #'test_arg_max_do_not_keepdims_example',
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
