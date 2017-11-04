from __future__ import division
import numpy as np
import pytest
from .ops_test_utils import _test_binary_op, AA, PRECISION_TO_TYPE,\
        unittest_helper

import cntk as C

def test_sigmoid(device_id):
    if device_id == -1:
        pytest.skip('Test only runs on GPU')

    a = C.input_variable((), dtype=np.float16, needs_gradient=True, name='a')
    s = C.sigmoid(a)
    result = s.eval([[0]])
    grad = s.grad([[0]])
    assert np.array_equal(result, np.asarray([0.5]).astype(np.float16))
    assert np.array_equal(grad, np.asarray([0.25]).astype(np.float16))

def test_cast(device_id):
    if device_id == -1:
        pytest.skip('Test only runs on GPU')
    i = C.input_variable((3))
    i2 = C.input_variable((1))
    i_data = [[1,20,300],[2000,30000,5000],[3,4,5]]
    i2_data = [[7],[8],[9]]
    f = C.combine(C.cast(i, dtype=np.float16), C.cast(i2, dtype=np.float16))
    data = f.eval({i:AA(i_data).astype(np.float32), i2:AA(i2_data).astype(np.float32)})
    assert np.array_equal(data[f[0]], i_data)
    assert np.array_equal(data[f[1]], i2_data)

def test_save_load(device_id, tmpdir):
    if device_id == -1:
        pytest.skip('Test only runs on GPU')
    i = C.input_variable((3), dtype='float16')
    t = C.times(i, C.parameter((3,5), dtype='float16', init=C.glorot_uniform()))
    data = AA([[1,2,3]]).astype(np.float16)
    result = t.eval(data)
    file = str(tmpdir / '1.dnn')
    t.save(file)
    t1 = C.load_model(file)
    result1 = t1.eval(data)
    assert np.array_equal(result, result1)