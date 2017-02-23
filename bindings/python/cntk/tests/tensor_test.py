# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from cntk import *
#from ..tensor import *
#from ..ops import constant, parameter
import numpy as np

#import pytest

def test_overload_exception():
    c = ops.constant(value=list(range(0, 10)))

    with pytest.raises(TypeError):
        c[:]

    with pytest.raises(TypeError):
        c[0:3:2]


def test_eval_scalar():
    c = ops.constant(value=2)
    assert (c+3).eval() == 5.0
    assert np.all((c+[3,4]).eval() == [5,6])

def test_numpy_conversion():

    ndav = sanitize_value((2,3), 1, np.float32, None)
    assert np.all(np.asarray(ndav) == np.ones((2,3)))

    # check Value
    assert np.all(np.asarray(Value(ndav)) == np.ones((2,3)))

    # check Constant
    c = ops.constant(1, shape=(2,3))
    assert np.all(np.asarray(c) == np.ones((2,3)))
    
    #check Parameter
    p = ops.parameter(shape=(2,3), init=1)
    assert np.all(np.asarray(p) == np.ones((2,3)))
    

if __name__=='__main__':
    #test_eval_scalar()
    test_numpy_conversion()
    test_overload_exception()
    print("tensor test passed")
