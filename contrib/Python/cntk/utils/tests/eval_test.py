
# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for eval() functionality that is used to discover and play with 
operations
"""

import numpy as np
import cntk

import pytest

_LEFT = [1., 2., 3., 4.]
_RIGHT = [1., 1., 0., 0.]
_EXPECTED = [np.asarray([[2., 3., 3., 4.]])]

def test_eval_plus():
    result = cntk.eval(cntk.plus(_LEFT, _RIGHT))
    TOLERANCE_ABSOLUTE = 1E-06    
    assert np.allclose(result, _EXPECTED, atol=TOLERANCE_ABSOLUTE)

def test_eval_plus_one_input():
    result = cntk.eval(cntk.plus(cntk.input_numpy([_LEFT]), _RIGHT))
    TOLERANCE_ABSOLUTE = 1E-06    
    assert np.allclose(result, _EXPECTED, atol=TOLERANCE_ABSOLUTE)

def test_eval_plus_one_input_last():
    result = cntk.eval(cntk.plus(_LEFT, cntk.input_numpy([_RIGHT])))
    TOLERANCE_ABSOLUTE = 1E-06    
    assert np.allclose(result, _EXPECTED, atol=TOLERANCE_ABSOLUTE)

def test_eval_plus_two_inputs():
    result = cntk.eval(cntk.plus(cntk.input_numpy([_LEFT]), cntk.input_numpy([_RIGHT])))
    TOLERANCE_ABSOLUTE = 1E-06    
    assert np.allclose(result, _EXPECTED, atol=TOLERANCE_ABSOLUTE)

def test_eval_plus_one_constant():
    result = cntk.eval(cntk.plus(cntk.constant(_LEFT), _RIGHT))
    TOLERANCE_ABSOLUTE = 1E-06    
    assert np.allclose(result, _EXPECTED, atol=TOLERANCE_ABSOLUTE)

def test_eval_plus_one_constant_last():
    result = cntk.eval(cntk.plus(_LEFT, cntk.constant(_RIGHT)))
    TOLERANCE_ABSOLUTE = 1E-06    
    assert np.allclose(result, _EXPECTED, atol=TOLERANCE_ABSOLUTE)

# this is dis-activated for now because we cannot have a netowrk without inputs
def _test_eval_plus_two_constants():
    result = cntk.eval(cntk.plus(cntk.constant(_LEFT), cntk.constant(_RIGHT)))
    TOLERANCE_ABSOLUTE = 1E-06    
    assert np.allclose(result, _EXPECTED, atol=TOLERANCE_ABSOLUTE)