# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

<<<<<<< 692095dd285683224403e986677fd5e145115fb2
<<<<<<< e09935f673f45670619bcbe7dbdaf774abfe7935

=======
from cntk import *
#from ..tensor import *
#from ..ops import constant, parameter
>>>>>>> fix compiler errors due to const initialization
=======
from ..tensor import *
from ..ops import constant, parameter
>>>>>>> add tests on ARM64 for ResNet_CIFAR10
import numpy as np
import cntk as C

import pytest

def test_overload_exception():
<<<<<<< 692095dd285683224403e986677fd5e145115fb2
    c = C.constant(value=list(range(0, 10)))
    with pytest.raises(ValueError):
=======
    c = constant(value=list(range(0, 10)))

    with pytest.raises(TypeError):
        c[:]

    with pytest.raises(TypeError):
>>>>>>> add tests on ARM64 for ResNet_CIFAR10
        c[0:3:2]

def test_eval_scalar():
<<<<<<< 692095dd285683224403e986677fd5e145115fb2
    c = C.constant(value=2)
=======
    c = constant(value=2)
>>>>>>> add tests on ARM64 for ResNet_CIFAR10
    assert (c+3).eval() == 5.0
    assert np.all((c+[3,4]).eval() == [5,6])

def test_numpy_conversion():
<<<<<<< 692095dd285683224403e986677fd5e145115fb2
    from cntk.internal import sanitize_value
=======
    from ..utils import sanitize_value
>>>>>>> add tests on ARM64 for ResNet_CIFAR10
    from ..cntk_py import Value

    # check NDArrayView
    ndav = sanitize_value((2,3), 1, np.float32, None)
    assert np.all(ndav.asarray() == np.ones((2,3)))

    # check Value
    assert np.all(Value(ndav).asarray() == np.ones((2,3)))

    # check Constant
<<<<<<< 692095dd285683224403e986677fd5e145115fb2
    c = C.constant(1, shape=(2,3))
    assert np.all(c.asarray() == np.ones((2,3)))
    
    #check Parameter
    p = C.parameter(shape=(2,3), init=1)
    assert np.all(p.asarray() == np.ones((2,3)))
=======
    c = constant(1, shape=(2,3))
    assert np.all(np.asarray(c) == np.ones((2,3)))
    
    #check Parameter
    p = parameter(shape=(2,3), init=1)
    assert np.all(np.asarray(p) == np.ones((2,3)))
    
>>>>>>> add tests on ARM64 for ResNet_CIFAR10
