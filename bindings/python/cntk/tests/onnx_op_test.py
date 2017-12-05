# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import numpy as np
import cntk as C
import pytest

#############
#helpers
#############
def save_validation_2inputs(x1, x2, y, file_path):
    constant_input1 = np.reshape(x1, (x1.size))
    constant_input2 = np.reshape(x2, (x2.size))
    constant_output = np.reshape(y, (y.size))
    validation_data = np.hstack((constant_input1, constant_input2, constant_output))
    c = C.input_variable((1))
    model = c * validation_data
    model.save(file_path, format=C.ModelFormat.ONNX)

def save_validation_1input(x,y, file_path):
    constant_input = np.reshape(x, (x.size))
    constant_output = np.reshape(y, (y.size))
    validation_data = np.hstack((constant_input, constant_output))
    c = C.input_variable((1))
    model = c * validation_data
    model.save(file_path, format=C.ModelFormat.ONNX)
    
    
def save_validation_no_input(y, file_path):
    constant_output = np.reshape(y, (y.size))
    c = C.input_variable((1))
    model = c * constant_output
    model.save(file_path, format=C.ModelFormat.ONNX)    

def verify_no_input(model, tmpdir, name):
    filename = os.path.join(str(tmpdir), name + R'.onnx')
    model.save(filename, format=C.ModelFormat.ONNX)

    loaded_model = C.Function.load(filename, format=C.ModelFormat.ONNX)
    assert model.shape == loaded_model.shape

    o = model.eval()
    o_ = loaded_model.eval()
    assert np.allclose(o_, o)

    validation_filename = os.path.join(str(tmpdir), name + R'_validation.onnx')
    save_validation_no_input(o, validation_filename)
    
def verify_one_input(model, data, tmpdir, name):
    filename = os.path.join(str(tmpdir), name + R'.onnx')
    model.save(filename, format=C.ModelFormat.ONNX)

    loaded_model = C.Function.load(filename, format=C.ModelFormat.ONNX)
    assert model.shape == loaded_model.shape

    o0 = model.eval({model.arguments[0]:data})
    o1 = loaded_model.eval({loaded_model.arguments[0]:data})

    assert np.allclose(o0, o1)

    validation_filename = os.path.join(str(tmpdir), name + R'_validation.onnx')
    save_validation_1input(data, o0, validation_filename)

def verify_two_input(model, data1, data2, tmpdir, name):
    filename = os.path.join(str(tmpdir), name + R'.onnx')
    model.save(filename, format=C.ModelFormat.ONNX)

    loaded_model = C.Function.load(filename, format=C.ModelFormat.ONNX)
    assert model.shape == loaded_model.shape

    o0 = model.eval({model.arguments[0]:data1, model.arguments[1]:data2})
    o1 = loaded_model.eval({loaded_model.arguments[0]:data1, loaded_model.arguments[1]:data2})

    assert np.allclose(o0, o1)
    
    validation_filename = os.path.join(str(tmpdir), name + R'_validation.onnx')
    save_validation_2inputs(data1, data2, o0, validation_filename)

#Abs
def test_Abs(tmpdir):
    shape = (4, 5)
    data = np.random.rand(*shape).astype(np.float32)

    model = C.abs(data)
    verify_no_input(model, tmpdir, 'Abs_0')

    x = C.input_variable(shape)
    model = C.abs(x)

    verify_one_input(model, data, tmpdir, 'Abs_1')

#Add
def test_Add(tmpdir):
    shape = (4, 5)
    data1 = np.random.rand(*shape).astype(np.float32)
    data2 = np.random.rand(*shape).astype(np.float32)
    model = C.plus(data1, data2)
    verify_no_input(model, tmpdir, 'Add_0')

    x = C.input_variable(shape)
    model = C.plus(x, data2)

    verify_one_input(model, data1, tmpdir, 'Add_1')

    y = C.input_variable(shape)
    model = C.plus(x, y)

    verify_two_input(model, data1, data2, tmpdir, 'Add_2')

#ArgMax
def test_ArgMax(tmpdir):
    shape = (4, 5)
    data = np.random.rand(*shape).astype(np.float32)
    model = C.argmax(data, 0)

    verify_no_input(model, tmpdir, 'ArgMax_0')

    x = C.input_variable(shape)
    model = C.argmax(x, 0)
    verify_one_input(model, data, tmpdir, 'ArgMax_1')

#ArgMin
def test_ArgMin(tmpdir):
    shape = (4, 5)
    data = np.random.rand(*shape).astype(np.float32)
    model = C.argmin(data, 0)

    verify_no_input(model, tmpdir, 'ArgMin_0')

#AveragePool
def test_AveragePool(tmpdir):
    img = np.reshape(np.arange(16, dtype = np.float32), [1, 4, 4])
    x = C.input_variable(img.shape)
    model = C.pooling(x, C.AVG_POOLING, (2,2), (2,2))

    verify_one_input(model, img, tmpdir, 'AveragePool')

#BatchNormalization
def test_BatchNormalization(tmpdir):
    dtype = np.float32

    sample = [  # 5 samples having 4 classes
        [1, 1, 2, 3],
        [0, 0, 0, 0],
        [3, 3, 4, 4],
        [1000, 1000, 1000, 1000],
        [10000, 10000, 10000, 10000]]

    epsilon = 0.00001

    t = np.asarray(sample, dtype=dtype).reshape(-1,1)
    mean = 1
    var = 2
    init_scale = 3
    init_bias = 4

    scale        = C.Parameter(init=np.asarray([init_scale], dtype=dtype), dtype=dtype)
    bias         = C.Parameter(init=np.asarray([init_bias], dtype=dtype), dtype=dtype)
    run_mean     = C.ops.constant(mean, shape=(1), dtype=dtype)
    run_variance = C.ops.constant(var,  shape=(1), dtype=dtype)
    run_count    = C.ops.constant(0,               dtype=dtype)

    a = C.input_variable(shape=(1), dtype=dtype, needs_gradient=False, name='a')

    op_node = C.batch_normalization(a, scale, bias, run_mean, run_variance, running_count=run_count, spatial=False,
        epsilon=epsilon)

    verify_one_input(op_node, t, tmpdir, 'BatchNormalization')

# Ceil
def test_Ceil(tmpdir):
    data = np.asarray([0.2, 1.3, 4., 5.5, 0.0], np.float32)
    model = C.ceil(data)

    verify_no_input(model, tmpdir, 'ceil_0')

    x = C.input_variable(data.shape)

    model = C.ceil(x)

    verify_one_input(model, data, tmpdir, 'ceil_1')

#Clip
def test_Clip(tmpdir):
    data = np.asarray([0.2, 1.3, 4., 5.5, 0.0], np.float32)
    min_v = 2
    max_v = 4
    model = C.clip(data, min_v, max_v)

    verify_no_input(model, tmpdir, 'clip_0')

    x = C.input_variable(data.shape)

    model = C.clip(x, min_v, max_v)

    verify_one_input(model, data, tmpdir, 'clip_1')

#Concat
def test_Concat(tmpdir):
    data1 = np.asarray([[[1, 2], [4, 5]]], dtype=np.float32)
    x = C.constant(value=data1)
    # create 3x2 matrix in a sequence of length 1 in a batch of one sample
    data2 = np.asarray([[[10, 20], 
                         [30, 40], 
                         [50, 60]]],dtype=np.float32)
    y = C.constant(value=data2)

    # splice both inputs on axis=0 returns a 5x2 matrix
    model = C.splice(x, y, axis=1)

    verify_no_input(model, tmpdir, 'Concat_0')

    x = C.input_variable(data1.shape)

    model = C.splice(x, y, axis=1)

    verify_one_input(model, data1, tmpdir, 'Concat__1')

#Div
def test_Div(tmpdir):
    data0 = np.asarray([1., 1., 1., 1.], dtype=np.float32)
    data1 = np.asarray([0.5, 0.25, 0.125, 0.], dtype=np.float32)
    model = C.element_divide(data0, data1)
    verify_no_input(model, tmpdir, 'Div_0')

    x = C.input_variable(data0.shape)
    model = C.element_divide(x, data1)
    verify_one_input(model, data0, tmpdir, 'Div_1')

#Dropout
def test_Dropout(tmpdir):
    data = np.asarray([[10, 20],[30, 40],[50, 60]], dtype=np.float32)
    model = C.dropout(data, 0.5)
    verify_no_input(model, tmpdir, 'Dropout_0')

    x = C.input_variable(data.shape)
    model = C.dropout(x, 0.5)
    verify_one_input(model, data, tmpdir, 'Dropout_1')

#Elu
def test_Elu(tmpdir):
    data = np.asarray([[-1, -0.5, 0, 1, 2]], dtype=np.float32)
    model = C.elu(data)
    verify_no_input(model, tmpdir, 'Elu_0')

    x = C.input_variable(data.shape)
    model = C.elu(x)
    verify_one_input(model, data, tmpdir, 'Elu_1')

#Equal
def test_Equal(tmpdir):
    data0 = np.asarray([41., 42., 43.], dtype=np.float32)
    data1 = np.asarray([42., 42., 42.], dtype=np.float32)
    model = C.equal(data0, data1)
    verify_no_input(model, tmpdir, 'Equal_0')

#Exp
def test_Exp(tmpdir):
    data = np.asarray([0., 1.], dtype=np.float32)
    model = C.exp(data)
    verify_no_input(model, tmpdir, 'Exp_0')

    x = C.input_variable(data.shape)
    model = C.exp(x)
    verify_one_input(model, data, tmpdir, 'Exp_1')

#Floor
def test_Floor(tmpdir):
    data = np.asarray([0.2, 1.3, 4., 5.5, 0.0], dtype=np.float32)
    model = C.floor(data)
    verify_no_input(model, tmpdir, 'Floor_0')

    x = C.input_variable(data.shape)
    model = C.floor(x)
    verify_one_input(model, data, tmpdir, 'Floor_1')

#Greater
def test_Greater(tmpdir):
    model = C.greater([41., 42., 43.], [42., 42., 42.])
    verify_no_input(model, tmpdir, 'Greater_0')

#Hardmax
def test_Hardmax(tmpdir):
    data = np.asarray([1., 1., 2., 3.], dtype=np.float32)
    model = C.hardmax(data)
    verify_no_input(model, tmpdir, 'Hardmax_0')

#LeakyRelu
def test_LeakyRelu(tmpdir):
    data = np.asarray([[-1, -0.5, 0, 1, 2]], dtype=np.float32)
    model = C.leaky_relu(data)
    verify_no_input(model, tmpdir, 'LeakyRelu_0')

#Less
def test_Less(tmpdir):
    data0 = np.asarray([41., 42., 43.], dtype=np.float32)
    data1 = np.asarray([42., 42., 42.], dtype=np.float32)

    model = C.less(data0, data1)
    verify_no_input(model, tmpdir, 'Less_0')

#Log
def test_Log(tmpdir):
    data = np.asarray([1., 2.], dtype=np.float32)
    model = C.log(data)
    verify_no_input(model, tmpdir, 'Log_0')

#LRN
def test_LRN(tmpdir):
    img_shape = (64, 32, 32)
    img = np.asarray(np.random.uniform(-1, 1, img_shape), dtype=np.float32)
    x_r = C.input_variable(shape=img_shape, dtype=np.float32)
    model = C.local_response_normalization(x_r, 2, 1.0, 0.0001, 0.75)
    verify_one_input(model, img, tmpdir, 'LRN_1')

#MatMul
def test_MatMul(tmpdir):
    data0 = np.asarray([[1,2],[3,4]], dtype=np.float32)
    data1 = np.asarray([[5],[6]], dtype=np.float32)
    model = C.times(data0, data1)
    verify_no_input(model, tmpdir, 'MatMul_0')

#Max
def test_Max(tmpdir):
    data0 = np.asarray([1., 1., 1., 1.], dtype=np.float32)
    data1 = np.asarray([0.5, 0.25, 0.125, 0.], dtype=np.float32)
    model = C.element_max(data0, data1)
    verify_no_input(model, tmpdir, 'Max_0')

#MaxPool
def test_MaxPool(tmpdir):
    img = np.reshape(np.arange(16, dtype = np.float32), [1, 4, 4])
    x = C.input_variable(img.shape)
    model = C.pooling(x, C.MAX_POOLING, (2,2), (3,3))
    verify_one_input(model, img, tmpdir, 'MaxPool_1')

#MaxRoiPool
def test_MaxRoiPool(tmpdir):
    input_map = [[[1., 2., 3.],       # (1, 3, 3) input operand (conv feature map)
           [4., 5., 6.],
           [7., 8., 9.]]]
    input_rois = [[1, 1, 2, 2]]

    conv_input = np.asarray(input_map, dtype=np.float32)
    roi_input = np.asarray(input_rois, dtype=np.float32)

    a = C.input_variable(shape=conv_input.shape,
                dtype=np.float32,
                needs_gradient=True,
                name='a')

    b = C.input_variable(shape=roi_input.shape,
                dtype=np.float32,
                needs_gradient=False,
                name='b')

    # adding batch and sequence axis
    conv_input.shape     = (1,) + conv_input.shape
    roi_input.shape      = (1,) + roi_input.shape

    model = C.roipooling(a, b, C.MAX_POOLING, (3,3), 1.)

    verify_two_input(model, conv_input, roi_input, tmpdir, 'MaxRoiPool_1')

#Min
def test_Min(tmpdir):
    data0 = np.asarray([1., 1., 1., 1.], dtype=np.float32)
    data1 = np.asarray([0.5, 0.25, 0.125, 0.], dtype=np.float32)
    model = C.element_min(data0, data1)
    verify_no_input(model, tmpdir, 'Min_0')

#Mul
def test_Mul(tmpdir):
    data0 = np.asarray([1., 1., 1., 1.], dtype=np.float32)
    data1 = np.asarray([0.5, 0.25, 0.125, 0.], dtype=np.float32)
    model = C.element_times(data0, data1)
    verify_no_input(model, tmpdir, 'ElementTimes_0')

#Neg
def test_Neg(tmpdir):
    data0 = np.asarray([1., -1., -2., 1.], dtype=np.float32)
    model = C.negate(data0)
    verify_no_input(model, tmpdir, 'Neg_0')

#PRelu
def test_PRelu(tmpdir):
    data = np.asarray([[-1, -0.5, 0, 1, 2]])
    alpha = C.constant(value=[[0.5, 0.5, 0.5, 0.5, 0.5]])
    model = C.param_relu(alpha, data)
    verify_no_input(model, tmpdir, 'PRelu_0')

#Pow
def test_Pow(tmpdir):
    model = C.pow([1, 2, -2], [3, -2, 3])
    verify_no_input(model, tmpdir, 'Pow_0')

#Reciprocal
def test_Reciprocal(tmpdir):
    model = C.reciprocal([-1/3, 1/5, -2, 3])
    verify_no_input(model, tmpdir, 'Reciprocal_0')

#ReduceLogSum
def test_ReduceLogSum(tmpdir):
    data = np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=np.float32)
    model = C.reduce_log_sum_exp(data, axis=0)

    verify_no_input(model, tmpdir, 'ReduceLogSum_0')

#ReduceMax
def test_ReduceMax(tmpdir):
    data = np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=np.float32)
    model = C.reduce_max(data, 0)
    verify_no_input(model, tmpdir, 'ReduceMax_0')

#ReduceMean
def test_ReduceMean(tmpdir):
    data = np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=np.float32)
    model = C.reduce_mean(data, 0)
    verify_no_input(model, tmpdir, 'ReduceMean_0')

#ReduceMin
def test_ReduceMin(tmpdir):
    data = np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=np.float32)
    model = C.reduce_min(data, 0)
    verify_no_input(model, tmpdir, 'ReduceMin_0')

#ReduceProd
def test_ReduceProd(tmpdir):
    data = np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=np.float32)
    model = C.reduce_prod(data, 0)
    verify_no_input(model, tmpdir, 'ReduceProd_0')

#ReduceSum
def test_ReduceSum(tmpdir):
    data = np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=np.float32)
    model = C.reduce_sum(data, 0)
    verify_no_input(model, tmpdir, 'ReduceSum_0')

#Relu
def test_Relu(tmpdir):
    data = [[-1, -0.5, 0, 1, 2]]
    model = C.relu([[-1, -0.5, 0, 1, 2]])
    verify_no_input(model, tmpdir, 'Relu_0')

#Reshape
def test_Reshape(tmpdir):
    data = np.asarray([[[[0., 1.],[2., 3.],[4., 5.]]]], dtype=np.float32)
    i1 = C.input_variable(shape=(3,2))
    model = C.reshape(i1, (2,3))
    verify_one_input(model, data, tmpdir, 'Reshape_1')

#Selu
def test_Selu(tmpdir):
    model = C.selu([[-1, -0.5, 0, 1, 2]])
    verify_no_input(model, tmpdir, 'Selu_0')

#Sigmoid
def test_Sigmoid(tmpdir):
    model = C.sigmoid([-2, -1., 0., 1., 2.])
    verify_no_input(model, tmpdir, 'Sigmoid_0')

#Slice
def test_Slice(tmpdir):
    data = np.asarray([[[1,2,-3], [4, 5, 6]]],dtype=np.float32)
    x1 = C.input_variable((2,3))
    model = C.slice(x1, 0, 1, 2)
    verify_one_input(model, data, tmpdir, 'Slice_0')

#Softmax
def test_Softmax(tmpdir):
    model = C.softmax([[1, 1, 2, 3]])
    verify_no_input(model, tmpdir, 'Softmax_0')

#Softplus
def test_Softplus(tmpdir):
    model = C.softplus([[-1, -0.5, 0, 1, 2]])
    verify_no_input(model, tmpdir, 'Softplus_0')

#Sqrt
def test_Sqrt(tmpdir):
    model = C.sqrt([0., 4.])
    verify_no_input(model, tmpdir, 'Sqrt_0')

#Sub
def test_Sub(tmpdir):
    model = C.minus([1, 2, 3], [4, 5, 6])
    verify_no_input(model, tmpdir, 'Sub_0')

#Tanh
def test_Tanh(tmpdir):
    model = C.tanh([[1,2],[3,4]])
    verify_no_input(model, tmpdir, 'Tanh_0')

#Transpose
def test_Transpose(tmpdir):
    a = np.arange(24).reshape(2,3,4).astype('f')
    model = C.transpose(a, perm=(2, 0, 1))
    verify_no_input(model, tmpdir, 'Transpose_0')



