# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from ..tensor import *
from ..ops import constant, parameter
import numpy as np

import pytest

def test_overload_exception():
    c = constant(value=list(range(0, 10)))

    with pytest.raises(ValueError):
        c[0:3:2]

def test_eval_scalar():
    c = constant(value=2)
    assert (c+3).eval() == 5.0
    assert np.all((c+[3,4]).eval() == [5,6])

def test_numpy_conversion():
    from cntk.internal import sanitize_value
    from ..cntk_py import Value

    # check NDArrayView
    ndav = sanitize_value((2,3), 1, np.float32, None)
    assert np.all(ndav.asarray() == np.ones((2,3)))

    # check Value
    assert np.all(Value(ndav).asarray() == np.ones((2,3)))

    # check Constant
    c = constant(1, shape=(2,3))
    assert np.all(c.asarray() == np.ones((2,3)))
    
    #check Parameter
    p = parameter(shape=(2,3), init=1)
    assert np.all(p.asarray() == np.ones((2,3)))

@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_ndarrayview_operators(device_id, precision):
    from scipy.special import expit
    from cntk.ops.tests.ops_test_utils import cntk_device

    def test(what, args, rtol=0, atol=0):
        args = [arg.astype(precision, copy=True) for arg in args]
        # TensorView
        res_tv = what(*[NDArrayView.from_dense(arg, device=cntk_device(device_id)) for arg in args]).to_ndarray()
        # numpy
        res_np = what(*args)
        print(res_tv)
        print(res_np)
        assert np.allclose(res_tv, res_np, rtol=rtol, atol=atol)

    x = np.array([[1., 2., 3.],[4., 5., 6.]])
    y = np.array([[13.],[42.]])

    # binary ops
    test(lambda a, b: a + b, [x, y])
    test(lambda a, b: a - b, [x, y])
    test(lambda a, b: a * b, [x, y])
    test(lambda a, b: a @ b, [y.reshape(1,2), x])
    test(lambda a, b: a.dot(b), [y.reshape(1,2), x])
    test(lambda a, b: a.dot_transpose(b) if isinstance(a, NDArrayView) else a.dot(b.transpose()), [y.reshape(1,2), x.transpose()])

    # unary ops
    test(lambda a: a.sigmoid() if isinstance(a, NDArrayView) else expit(a), [x], rtol=1e-6)
    test(lambda a: a.tanh() if isinstance(a, NDArrayView) else np.tanh(a), [x], rtol=1e-6)
    test(lambda a: a.relu() if isinstance(a, NDArrayView) else np.maximum(a,0), [x])

    # reduction ops
    test(lambda a: a.reduce_log_sum() if isinstance(a, NDArrayView) else np.log(np.sum(np.exp(a))), [x], rtol=1e-6)

    # reshape
    test(lambda a: a.reshape((1,6)), [x])

    # slice
    test(lambda a: a[:], [x])
    test(lambda a: a[:1], [x])
    test(lambda a: a[1:,:], [x])
    #test(lambda a: a[1,1], [x]) # BUGBUG: This should work
    test(lambda a: a[:2,:], [x])
    test(lambda a: a[1:2,:], [x])
    test(lambda a: a[...,:], [x])
    def atest(a,b):
        a[1:2,:] = b
        return a
    test(atest, [x, np.array(13)])

    # in-place ops
    test(lambda a, b: a.__iadd__(b), [x, y]) 
    test(lambda a, b: a.__isub__(b), [x, y]) 
