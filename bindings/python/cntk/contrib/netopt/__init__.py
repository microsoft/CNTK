# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""
Netowrk optimization alogorithms.
"""
import sys
import cntk as C


def try_register_native_convolve_function():
    '''
    Register the native binary convolution function that calls halide
    operations internally.
    '''
    try:
        C.ops.register_native_user_function(
                    'NativeBinaryConvolveFunction', 
                    'Cntk.BinaryConvolution-' + C.__version__.rstrip('+'), 
                    'CreateBinaryConvolveFunction')
        native_convolve_function_registered = True
    except:
        native_convolve_function_registered = False
    
    module = sys.modules[__name__]   
    setattr(module, 'native_convolve_function_registered', native_convolve_function_registered)

try_register_native_convolve_function()