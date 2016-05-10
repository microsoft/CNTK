
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

def test_eval_plus():
    result = cntk.eval(cntk.plus([1., 2., 3., 4.], [1., 1., 0., 0.]))
    TOLERANCE_ABSOLUTE = 1E-06    
    assert np.allclose(result, np.asarray([2., 3., 3., 4.]), atol=TOLERANCE_ABSOLUTE)

def test_eval_plus_one_input():
    result = cntk.eval(cntk.plus(cntk.input_numpy([[1., 2., 3., 4.]]), [1., 1., 0., 0.]))
    TOLERANCE_ABSOLUTE = 1E-06    
    assert np.allclose(result, np.asarray([2., 3., 3., 4.]), atol=TOLERANCE_ABSOLUTE)

def test_eval_plus_one_input_last():
    result = cntk.eval(cntk.plus([1., 2., 3., 4.], cntk.input_numpy([[1., 1., 0., 0.]])))
    TOLERANCE_ABSOLUTE = 1E-06    
    assert np.allclose(result, np.asarray([2., 3., 3., 4.]), atol=TOLERANCE_ABSOLUTE)

def test_eval_plus_two_inputs():
    result = cntk.eval(cntk.plus(cntk.input_numpy([[1., 2., 3., 4.]]), cntk.input_numpy([[1., 1., 0., 0.]])))
    TOLERANCE_ABSOLUTE = 1E-06    
    assert np.allclose(result, np.asarray([2., 3., 3., 4.]), atol=TOLERANCE_ABSOLUTE)
	
def test_eval_plus_one_constant():
    result = cntk.eval(cntk.plus(cntk.constant([1., 2., 3., 4.]), [1., 1., 0., 0.]))
    TOLERANCE_ABSOLUTE = 1E-06    
    assert np.allclose(result, np.asarray([2., 3., 3., 4.]), atol=TOLERANCE_ABSOLUTE)	
	
def test_eval_plus_one_constant_last():
    result = cntk.eval(cntk.plus([1., 2., 3., 4.], cntk.constant([1., 1., 0., 0.])))
    TOLERANCE_ABSOLUTE = 1E-06    
    assert np.allclose(result, np.asarray([2., 3., 3., 4.]), atol=TOLERANCE_ABSOLUTE)		

# this is dis-activated for now because we cannot have a netowrk without inputs	
def _test_eval_plus_two_constants():
    result = cntk.eval(cntk.plus(cntk.constant([1., 2., 3., 4.]), cntk.constant([1., 1., 0., 0.])))
    TOLERANCE_ABSOLUTE = 1E-06    
    assert np.allclose(result, np.asarray([2., 3., 3., 4.]), atol=TOLERANCE_ABSOLUTE)			