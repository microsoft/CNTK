# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import cntk as C
import numpy as np

def _check(expected, d):
    for key in expected:
        assert key in d
        assert d[key] == expected[key]
    for key in d:
        assert key in expected

def test_convolution_attributes():
    x = C.input_variable( (1, 5, 5) )
    filter = np.reshape(np.array([2, -1, -1, 2], dtype = np.float32), (1, 2, 2))
    kernel = C.constant(value = filter)
    f = C.convolution(kernel , x, auto_padding = [False])
    d = f.root_function.attributes
    expected = {'autoPadding': [False, False, False], 
        'sharing': [True, True, True], 
        'groups': 1,
        'strides': (1, 1, 1), 
        'dilation': (1, 1, 1),
        'maxTempMemSizeInSamples': 0, 
        'upperPad': (0, 0, 0), 
        'lowerPad': (0, 0, 0),
        'transpose': False,
        'outputShape': (0,),
        'kernelShape': (1, 2, 2),
        'sequential': False
        }
    _check(expected, d)

    f = C.convolution(kernel , x, auto_padding = [False, True])
    d = f.root_function.attributes
    expected = {'autoPadding': [False, False, True], 
        'sharing': [True, True, True], 
        'groups': 1,
        'strides': (1, 1, 1),
        'dilation': (1, 1, 1),
        'maxTempMemSizeInSamples': 0, 
        'upperPad': (0, 0, 0), 
        'lowerPad': (0, 0, 0),
        'transpose': False,
        'outputShape': (0,),
        'kernelShape': (1, 2, 2),
        'sequential': False
        }
    _check(expected, d)

def test_convolution_transpose_attributes():
    x = C.input_variable( (1, 5, 5) )
    filter = np.reshape(np.array([2, -1, -1, 2], dtype = np.float32), (1, 2, 2))
    kernel = C.constant(value = filter)
    f = C.convolution_transpose(kernel , x, auto_padding = [False])
    d = f.root_function.attributes
    expected = {'autoPadding': [False, False, False], 
        'sharing': [True, True, True], 
        'groups': 1,
        'strides': (1, 1, 1),
        'dilation': (1, 1, 1),
        'maxTempMemSizeInSamples': 0, 
        'upperPad': (0, 0, 0), 
        'lowerPad': (0, 0, 0),
        'transpose': True,
        'outputShape': (0,),
        'kernelShape': (1, 2, 2),
        'sequential': False
        }
    _check(expected, d)

def test_dropout_attributes():
    x = C.input_variable( (1, 5, 5) )
    f = C.dropout(x, 0.5, 42)
    d = f.root_function.attributes
    expected = {'dropoutRate': 0.5, 'rngSeed' : 42, 'rngOffset' : 0}
    _check(expected, d)

def test_slice_attributes():
    x = C.input_variable((2,3))
    f = C.slice(x, 0, 1, 2)
    d = f.root_function.attributes
    expected = {'endIndex': 2, 'beginIndex': 1, 'axis': ('ordered', 'static', 1), 'sliceStrides': 1}
    _check(expected, d)
    f = C.slice(x, [0,1], [1,0], [2,2], [-1,1])
    d = f.root_function.attributes
    expected = {'endIndexVec': [2,2], 'beginIndexVec': [1,0], 'axisVec': [('ordered', 'static', 1), ('ordered', 'static', 0)], 'sliceStridesVec': [-1, 1]}
    _check(expected, d)
