# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================


import numpy as np
import cntk as C

import pytest

def test_overload_exception():
    c = C.constant(value=list(range(0, 10)))

    with pytest.raises(ValueError):
        c[0:3:2]

def test_eval_scalar():
    c = C.constant(value=2)
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
    c = C.constant(1, shape=(2,3))
    assert np.all(c.asarray() == np.ones((2,3)))
    
    #check Parameter
    p = C.parameter(shape=(2,3), init=1)
    assert np.all(p.asarray() == np.ones((2,3)))

@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_ndarrayview_operators(device_id, precision):
    from scipy.special import expit
    from cntk.ops.tests.ops_test_utils import cntk_device
    from cntk.ops import input_variable, sigmoid, tanh, relu, exp, reduce_sum, reduce_log_sum_exp, reshape
    from ..cntk_py import Variable

    def test(what, args, rtol=0, atol=0):
        args = [arg.astype(precision, copy=True) for arg in args]
        # TensorView
        args_tv = [NDArrayView.from_dense(arg) for arg in args]
        res_tv = what(*args_tv)
        assert isinstance(res_tv, NDArrayView) # make sure we don't get a cntk_py version back  --TODO: figure out why this does not work
        res_tv = res_tv.to_ndarray()
        ## known value
        #args_kv = [constant(arg) for arg in args]
        #res_kv = what(*args_kv)
        #res_kv = res_kv.value().to_ndarray()
        #assert np.allclose(res_tv, res_kv, rtol=rtol, atol=atol)
        ## static graph
        #inputs_v2 = [input_variable(arg.shape) for arg in args]
        #f_v2 = what(*inputs_v2)
        #res_v2 = f_v2(*args)[0]
        #assert np.allclose(res_tv, res_v2, rtol=rtol, atol=atol)
        # numpy
        res_np = what(*args)
        print(res_tv)
        print(res_np)
        assert np.allclose(res_tv, res_np, rtol=rtol, atol=atol)

    mat23 = np.array([[1., 2., 3.],[4., 5., 6.]]) # 3 x 2 matrix
    col21 = np.array([[13.],[42.]])                # column vector, same height as matrix

    def is_not_numpy(a):
        return isinstance(a, (NDArrayView, Variable))
    def is_data(a):
        return isinstance(a, NDArrayView)
    def is_var(a):
        return isinstance(a, Variable)

    # binary ops, elementwise
    test(lambda a, b: a + b, [mat23, col21])
    test(lambda a, b: a + b, [mat23, np.array(13)]) # scalar
    test(lambda a, b: a - b, [mat23, col21])
    test(lambda a, b: a * b, [mat23, col21])

    # matrix product
    test(lambda a, b: a @ b, [col21.reshape(1,2), mat23])
    test(lambda a, b: a.dot(b), [col21.reshape(1,2), mat23])
    test(lambda a, b: a.dot_transpose(b) if is_not_numpy(a) else a.dot(b.transpose()), [col21.reshape(1,2), mat23.transpose()])
    test(lambda a, b: a.dot_transpose(b) if is_not_numpy(a) else a.dot(b.transpose()), [mat23, mat23]) # mat23 * mat23^T
    test(lambda a, b: a.dot_transpose(b) if is_not_numpy(a) else a.dot(b.transpose()), [mat23.transpose(), mat23.transpose()]) # mat23^T * mat23
    test(lambda a, b: a.dot_transpose(b) if is_not_numpy(a) else a.dot(b.transpose()), [mat23, np.array([13.,42.,1968.])]) # mat23 * row3^T
    test(lambda a, b: a.dot_transpose(b) if is_not_numpy(a) else a.dot(b.transpose()), [np.array([13.,42.,1968.]), mat23]) # row3 * mat23^T
    test(lambda a, b: a.transpose_dot(b) if is_not_numpy(a) else a.transpose().dot(b), [mat23, mat23]) # mat23^T * mat23
    test(lambda a, b: a.transpose_dot(b) if is_not_numpy(a) else a.transpose().dot(b), [col21.reshape(2), mat23])
    test(lambda a, b: a.transpose_dot(b) if is_not_numpy(a) else a.transpose().dot(b), [col21, mat23])

    # unary ops
    test(lambda a: a.sigmoid() if is_data(a) else sigmoid(a) if is_var(a) else expit(a),        [mat23], rtol=1e-6)
    test(lambda a: a.tanh()    if is_data(a) else tanh(a)    if is_var(a) else np.tanh(a),      [mat23], rtol=1e-6)
    test(lambda a: a.relu()    if is_data(a) else relu(a)    if is_var(a) else np.maximum(a,0), [mat23])
    test(lambda a: a.exp()     if is_data(a) else exp(a)     if is_var(a) else np.exp(a),       [mat23], rtol=1e-6)

    # reduction ops
    test(lambda a: a.reduce_sum()     if is_data(a) else reduce_sum(a)         if is_var(a) else np.sum(a),                 [mat23], rtol=1e-6)
    test(lambda a: a.reduce_log_sum() if is_data(a) else reduce_log_sum_exp(a) if is_var(a) else np.log(np.sum(np.exp(a))), [mat23], rtol=1e-6)

    # reshape
    test(lambda a: a.reshape((1,6)) if not is_var(a) else reshape(a, (1,6)), [mat23])

    # slice
    test(lambda a: a[:],     [mat23])
    test(lambda a: a[:1],    [mat23])
    test(lambda a: a[1:,:],  [mat23])
    test(lambda a: a[1,1],   [mat23])
    test(lambda a: a[1:2,1], [mat23])
    test(lambda a: a[:2,:],  [mat23])
    test(lambda a: a[1:2,:], [mat23])
    test(lambda a: a[...,:], [mat23])
    def atest(a,b):
        a[1:2,:] = b
        return a
    test(atest, [mat23, np.array(13)])
    def itest(a):
        for x in a:
            x[:] = x + x
        return a
    test(itest, [mat23]) # test loop over first index, IndexError

    # splice
    test(lambda *args: NDArrayView.splice(*args) if is_not_numpy(args[0]) else np.concatenate((args[0][np.newaxis,:], args[1][np.newaxis,:])), [mat23, 1.1*mat23])

    # in-place ops
    test(lambda a, b: a.__iadd__(b), [mat23, col21]) 
    test(lambda a, b: a.__isub__(b), [mat23, col21]) 
