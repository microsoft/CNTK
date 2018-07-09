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
    'bvlc_alexnet',
    'densenet121',
    'inception_v1',
    'inception_v2',
    'resnet50',
    'shufflenet',
    'vgg16',
    'vgg19',
    'zfnet512',  # Attempt to use DefaultLogger but none has been registered.
    'PixelShuffle'
]

skip_ops = [
    # These tests crash and prevent tests after them from running.
    'test_reshape_extended_dims_cpu',
    'test_reshape_extended_dims_cuda',
    'test_reshape_negative_dim_cpu',
    'test_reshape_negative_dim_cuda',
    'test_reshape_one_dim_cpu',
    'test_reshape_one_dim_cuda',
    'test_reshape_reduced_dims_cpu',
    'test_reshape_reduced_dims_cuda',
    'test_reshape_reordered_dims_cpu',
    'test_reshape_reordered_dims_cuda',
    'test_pow_bcast_scalar_cpu',
    'test_pow_bcast_scalar_cuda',
    'test_operator_repeat_dim_overflow_cpu',
    'test_operator_repeat_dim_overflow_cuda',

    # These use pre-ONNX 1.2 incompatible versions of the operators.
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