from __future__ import division
import numpy as np
import pytest
from .ops_test_utils import _test_binary_op, AA, PRECISION_TO_TYPE,\
        unittest_helper

import cntk as C

def test_sigmoid():
    a = C.input_variable((), dtype=np.float16, needs_gradient=True, name='a')
    s = C.sigmoid(a)
    result = s.eval([[0]])
    grad = s.grad([[0]])
    assert np.array_equal(result, np.asarray([0.5]).astype(np.float16))
    assert np.array_equal(grad, np.asarray([0.25]).astype(np.float16))

def test_cast():
    i = C.input_variable((3))
    i2 = C.input_variable((1), needs_gradient=True)
    i_data = [[1,20,300],[2000,3000,5000],[3,4,5]]
    i2_data = [[7],[8],[9]]
    f = C.combine(C.cast(i, dtype=np.float16), C.cast(i2, dtype=np.float16))
    feed_dict = {i:AA(i_data).astype(np.float32), i2:AA(i2_data).astype(np.float32)}
    data = f.eval(feed_dict)
    assert np.array_equal(data[f[0]], i_data)
    assert np.array_equal(data[f[1]], i2_data)
    s = f[0] * f[1]
    data = s.grad(feed_dict)
    assert np.array_equal(data, [[321],[10000],[12]])

def test_save_load(tmpdir):
    i = C.input_variable((3), dtype='float16')
    t = C.times(i, C.parameter((3,5), dtype='float16', init=C.glorot_uniform()))
    data = AA([[1,2,3]]).astype(np.float16)
    result = t.eval(data)
    file = str(tmpdir / '1.dnn')
    t.save(file)
    t1 = C.load_model(file)
    result1 = t1.eval(data)
    assert np.array_equal(result, result1)

def test_batchnorm(device_id):
    if device_id == -1:
        pytest.skip('Test only runs on GPU')
    shape = (3,)
    i = C.input_variable(shape, dtype='float16')
    scale = C.parameter(shape, init=1, dtype='float')
    bias = C.parameter(shape, init=2, dtype='float')
    run_mean = C.constant(3, shape=shape, dtype='float')
    run_variance = C.constant(4, shape=shape, dtype='float')
    run_count = C.constant(0, shape=(), dtype='float')
    
    bn = C.batch_normalization(i, scale, bias, run_mean, run_variance, running_count=run_count,
                                   spatial=False, normalization_time_constant=5000, blend_time_constant=0, epsilon=0.00001,
                                   use_cudnn_engine=True, disable_regularization=True)

    data = AA([[1,2,3]]).astype(np.float16)
    bn.grad(data, wrt=[scale,bias])

def test_rnn(device_id):
    if device_id == -1:
        pytest.skip('Test only runs on GPU')

    batch_size = 8
    sequence_len = 100
    vocab_dim = 20
    embed_dim = 10
    hidden_dim = 7
    input = C.cast(C.sequence.input_variable(()), np.float16)
    with C.default_options(dtype=np.float16):
        embed = C.layers.Embedding(embed_dim)(C.one_hot(input, num_classes=vocab_dim, sparse_output=False))
        z = C.layers.Recurrence(C.layers.LSTM(hidden_dim))(embed)

    feed = np.floor(np.random.rand(batch_size, sequence_len).astype(np.float32) * (vocab_dim - 1))
    z.grad(feed, wrt=z.parameters)
    
    num_layers = 2
    W = C.parameter((C.InferredDimension, embed_dim), init=C.glorot_uniform(), dtype=np.float16)
    with C.default_options(dtype=np.float16):
        z = C.optimized_rnnstack(embed, W, hidden_dim, num_layers)

    feed = np.floor(np.random.rand(batch_size, sequence_len).astype(np.float32) * (vocab_dim - 1))
    z.grad(feed, wrt=z.parameters)
