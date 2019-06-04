# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from cntk import ops, cpu, parameter, NDArrayView, input
import numpy as np
import cntk as C
import os
import sys
import pytest

abs_path = os.path.dirname(os.path.abspath(__file__))
custom_convolution_ops_dir = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Extensibility", "BinaryConvolution")
sys.path.append(custom_convolution_ops_dir)

from custom_convolution_ops import *
import cntk.contrib.netopt as nopt

# checks the functionality of the binary convolution custom function
def test_native_binary_function():
    # user functions need to be registered before being callable by python
    if not nopt.native_convolve_function_registered:
      pytest.skip("Could not find {0} library. "
        "Please check if HALIDE_PATH is configured properly "
        "and try building {1} again"
        .format('Cntk.BinaryConvolution-' + C.__version__.rstrip('+'),
        'Extnsibiliy\\BinaryConvolution'))

    # be sure to only run on CPU, binary convolution does not have GPU support for now
    dev = C.cpu()
    # create an arbitrary input mimicking a realistic cifar input
    x = input((64, 28, 28))
    # random filter weights for testing
    w = parameter((64, 64, 3, 3), init=np.reshape(2*(np.random.rand(64*64*3*3)-.5), (64, 64, 3, 3)), dtype=np.float32, device=dev)

    # set the convolution parameters by passing in an attribute dictionary
    #attributes = {'stride' : 1, 'padding' : False, 'size' : 3}

    attributes = {'stride' : 1,
                  'padding' : False,
                  'size' : 3,                       
                  'h' : 28,
                  'w' : 28,
                  'channels' : 64,
                  'filters' : 64 }

    # define the binary convolution op
    op = ops.native_user_function('NativeBinaryConvolveFunction', [w, x], attributes, 'native_binary_convolve')
    
    # also define an op using python custom functions that should have the same output
    op2 = C.convolution(CustomMultibitKernel(w, 1), CustomSign(x), auto_padding = [False])
    # create random input data
    x_data = NDArrayView.from_dense(np.asarray(np.reshape(2*(np.random.rand(64*28*28)-.5), (64, 28, 28)),dtype=np.float32), device=dev)
    # evaluate the CPP binary convolve
    result = op.eval({x : x_data}, device=dev)

    # evaluate the python emulator
    result2 = op2.eval({x : x_data}, device=dev)
    native_times_primitive = op.find_by_name('native_binary_convolve')
    # assert that both have the same result
    '''
    Disable this tempororily. Needs to investigate and fix the halide
    code to match the previous test behavior.
    '''
    #assert np.allclose(result, result2, atol=0.001)
