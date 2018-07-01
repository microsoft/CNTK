# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import numpy as np
import cntk as C
import pytest
from cntk.ops.tests.ops_test_utils import cntk_device
from itertools import product

#############
#helpers
#############
def verify_no_input(model, tmpdir, name):
    filename = os.path.join(str(tmpdir), name + R'.onnx')
    model.save(filename, format=C.ModelFormat.ONNX)

    loaded_model = C.Function.load(filename, format=C.ModelFormat.ONNX)
    assert model.shape == loaded_model.shape

    o = model.eval()
    o_ = loaded_model.eval()
    assert np.allclose(o_, o)
    
def verify_one_input(model, data, tmpdir, name, device=None):
    filename = os.path.join(str(tmpdir), name + R'.onnx')
    model.save(filename, format=C.ModelFormat.ONNX)

    loaded_model = C.Function.load(filename, format=C.ModelFormat.ONNX)
    assert model.shape == loaded_model.shape

    if device:
        o0 = model.eval({model.arguments[0]:data}, device=device)
        o1 = loaded_model.eval({loaded_model.arguments[0]:data}, device=device)
    else:
        o0 = model.eval({model.arguments[0]:data})
        o1 = loaded_model.eval({loaded_model.arguments[0]:data})

    if (type(o0) is list):
        o0 = o0[0]
    if (type(o1) is list):
        o1 = o1[0]

    assert np.allclose(o0, o1)

def verify_two_input(model, data1, data2, tmpdir, name):
    filename = os.path.join(str(tmpdir), name + R'.onnx')
    model.save(filename, format=C.ModelFormat.ONNX)

    loaded_model = C.Function.load(filename, format=C.ModelFormat.ONNX)
    assert model.shape == loaded_model.shape

    o0 = model.eval({model.arguments[0]:data1, model.arguments[1]:data2})
    o1 = loaded_model.eval({loaded_model.arguments[0]:data1, loaded_model.arguments[1]:data2})

    assert np.allclose(o0, o1)

#Shared Test Configs
DType_Config = (np.float32, np.float16)
    
#Abs
@pytest.mark.parametrize("dtype", DType_Config)
def test_Abs(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        shape = (4, 5)
        data = np.random.rand(*shape).astype(dtype)

        model = C.abs(data)
        verify_no_input(model, tmpdir, 'Abs_0')

        x = C.input_variable(shape)
        model = C.abs(x)

        verify_one_input(model, data, tmpdir, 'Abs_1')

#Add
@pytest.mark.parametrize("dtype", DType_Config)
def test_Add(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        shape = (4, 5)
        data1 = np.random.rand(*shape).astype(dtype)
        data2 = np.random.rand(*shape).astype(dtype)
        model = C.plus(data1, data2)
        verify_no_input(model, tmpdir, 'Add_0')

        x = C.input_variable(shape)
        model = C.plus(x, data2)

        verify_one_input(model, data1, tmpdir, 'Add_1')

        y = C.input_variable(shape)
        model = C.plus(x, y)

        verify_two_input(model, data1, data2, tmpdir, 'Add_2')

#And
def test_And(tmpdir):
    pytest.skip('Need to support new ONNX spec.')
    data1 = np.asarray([[1, 1, 0, 0],[1, 1, 1, 1]], dtype)
    data2 = np.asarray([1, 0, 1, 0], dtype)

    model = C.element_and(data1, data2)
    verify_no_input(model, tmpdir, 'And_0')

    x = C.input_variable(np.shape(data1))
    y = C.input_variable(np.shape(data2))

    model = C.element_and(x, data2)
    verify_one_input(model, data1, tmpdir, 'And_1')

    model = C.element_and(x, y)
    verify_two_input(model, data1, data2, tmpdir, 'And_2')

#Or
def test_Or(tmpdir):
    pytest.skip('Need to support new ONNX spec.')
    data1 = np.asarray([[1, 1, 0, 0],[1, 1, 1, 1]], np.float32)
    data2 = np.asarray([1, 0, 1, 0], np.float32)

    model = C.element_or(data1, data2)
    verify_no_input(model, tmpdir, 'Or_0')

    x = C.input_variable(np.shape(data1))
    y = C.input_variable(np.shape(data2))

    model = C.element_or(x, data2)
    verify_one_input(model, data1, tmpdir, 'Or_1')

    model = C.element_or(x, y)
    verify_two_input(model, data1, data2, tmpdir, 'Or_2')

#Xor
def test_Xor(tmpdir):
    pytest.skip('Need to support new ONNX spec.')
    data1 = np.asarray([[1, 1, 0, 0],[1, 1, 1, 1]], np.float32)
    data2 = np.asarray([1, 0, 1, 0], np.float32)

    model = C.element_xor(data1, data2)
    verify_no_input(model, tmpdir, 'Xor_0')

    x = C.input_variable(np.shape(data1))
    y = C.input_variable(np.shape(data2))

    model = C.element_xor(x, data2)
    verify_one_input(model, data1, tmpdir, 'Xor_1')

    model = C.element_xor(x, y)
    verify_two_input(model, data1, data2, tmpdir, 'Xor_2')

#Not
@pytest.mark.parametrize("dtype", DType_Config)
def test_Not(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data1 = np.asarray([[1, 1, 0, 0],[1, 1, 1, 1]]).astype(dtype)

        model = C.element_not(data1)
        verify_no_input(model, tmpdir, 'Not_0')

        x = C.input_variable(np.shape(data1))

        model = C.element_not(x)
        verify_one_input(model, data1, tmpdir, 'Not_1')

#ArgMax
@pytest.mark.parametrize("dtype", DType_Config)
def test_ArgMax(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        shape = (4, 5)
        data = np.random.rand(*shape).astype(dtype)
        model = C.argmax(data, 0)

        verify_no_input(model, tmpdir, 'ArgMax_0')

        x = C.input_variable(shape)
        model = C.argmax(x, 0)
        verify_one_input(model, data, tmpdir, 'ArgMax_1')

#ArgMin
@pytest.mark.parametrize("dtype", DType_Config)
def test_ArgMin(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        shape = (4, 5)
        data = np.random.rand(*shape).astype(dtype)
        model = C.argmin(data, 0)

        verify_no_input(model, tmpdir, 'ArgMin_0')

#AveragePool
@pytest.mark.parametrize("dtype", DType_Config)
def test_AveragePool(tmpdir, dtype, device_id):
    if device_id == -1 and dtype == np.float16:
        pytest.skip('Test is skipped on CPU with float16 data')
    device = cntk_device(device_id)
    with C.default_options(dtype=dtype):
        img = np.reshape(np.arange(16, dtype = dtype), [1, 4, 4])
        x = C.input_variable(img.shape)
        model = C.pooling(x, C.AVG_POOLING, (2,2), (2,2))

        verify_one_input(model, img, tmpdir, 'AveragePool_1', device)

#BatchNormalization
@pytest.mark.parametrize("dtype", DType_Config)
def test_BatchNormalization(tmpdir, dtype):
    if (dtype == np.float16):
        pytest.skip("TO BE FIXED")
    with C.default_options(dtype = dtype):
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
@pytest.mark.parametrize("dtype", DType_Config)
def test_Ceil(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.asarray([0.2, 1.3, 4., 5.5, 0.0], dtype)
        model = C.ceil(data)

        verify_no_input(model, tmpdir, 'ceil_0')

        x = C.input_variable(data.shape)

        model = C.ceil(x)

        verify_one_input(model, data, tmpdir, 'ceil_1')

#Clip
@pytest.mark.parametrize("dtype", DType_Config)
def test_Clip(tmpdir, dtype):
    if (dtype == np.float16):
        pytest.skip("TO BE FIXED")
    with C.default_options(dtype = dtype):
        data = np.asarray([0.2, 1.3, 4., 5.5, 0.0], dtype)
        min_v = 2
        max_v = 4
        model = C.clip(data, min_v, max_v)

        verify_no_input(model, tmpdir, 'clip_0')

        x = C.input_variable(data.shape)

        model = C.clip(x, min_v, max_v)

        verify_one_input(model, data, tmpdir, 'clip_1')

#Concat
@pytest.mark.parametrize("dtype", DType_Config)
def test_Concat(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data1 = np.asarray([[[1, 2], [4, 5]]], dtype=dtype)
        x = C.constant(value=data1)
        # create 3x2 matrix in a sequence of length 1 in a batch of one sample
        data2 = np.asarray([[[10, 20], 
                             [30, 40], 
                             [50, 60]]],dtype=dtype)
        y = C.constant(value=data2)

        # splice both inputs on axis=0 returns a 5x2 matrix
        model = C.splice(x, y, axis=1)

        verify_no_input(model, tmpdir, 'Concat_0')

        x = C.input_variable(data1.shape)

        model = C.splice(x, y, axis=1)

        verify_one_input(model, data1, tmpdir, 'Concat_1')

@pytest.mark.parametrize("dtype", DType_Config)
def test_ConvTranspose(tmpdir, dtype, device_id):
    if device_id == -1 and dtype == np.float16:
        pytest.skip('Test is skipped on CPU with float16 data')
    device = cntk_device(device_id)
    with C.default_options(dtype=dtype):
        # Keep the shapes below as they are, because this tests an earlier bug.
        input_shape = (48, 16, 16) 
        img = np.reshape(np.arange(np.prod(input_shape), dtype = dtype), input_shape) 

        x = C.input_variable(input_shape)

        kernel_shape = (48, 32, 3, 3) # For convolution_transpose the shape is (I x O x W x H)
        kernel = C.constant(value = np.ones(shape=(kernel_shape), dtype = dtype))

        conv_trans_model = C.convolution_transpose(kernel, x, strides=(2, 2), output_shape=(32, 32, 32), auto_padding = [False, True, True])

        verify_one_input(conv_trans_model, img, tmpdir, 'ConvTranspose_0', device)

# DepthToSpace
@pytest.mark.parametrize("dtype", DType_Config)
def test_DepthToSpace(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        num_channels = 9
        block_size = 3
        image_shape = (4, 5)
        input_val = np.array(np.reshape(range(num_channels), (num_channels, 1, 1)), dtype=dtype)
        input_val = np.tile(input_val, (1,) + image_shape)
        input_val.shape = (1,) + input_val.shape
        img = C.input_variable((num_channels,) + image_shape, dtype=dtype)
        model = C.depth_to_space(img, block_size)

        verify_one_input(model, input_val, tmpdir, 'DepthToSpace')

#Div
def test_Div(tmpdir):
    pytest.skip('Need to support new ONNX spec.')
    def run_div_test(shape1, shape2, tmpdir):
        broadcast = 'no_broadcast'
        if (shape1 != shape2):
            broadcast = 'with_broadcast'

        data1 = np.random.rand(*shape1).astype(np.float32)
        data2 = np.random.rand(*shape2).astype(np.float32)

        x = C.input_variable(shape1)
        y = C.input_variable(shape2)

        model = C.element_divide(data1, data2)
        verify_no_input(model, tmpdir, 'Div_' + broadcast + '_d1d2')

        model = C.element_divide(x, data2)
        verify_one_input(model, data1, tmpdir, 'Div_' + broadcast + '_xd2')

        model = C.element_divide(data1, y)
        verify_one_input(model, data2, tmpdir, 'Div_' + broadcast + '_d1y')

        model = C.element_divide(x, y)
        verify_two_input(model, data1, data2, tmpdir, 'Div_' + broadcast + '_xy')

    shape1 = (2, 3, 4, 5)
    shape2 = shape1
    # without broadcast
    run_div_test(shape1, shape2, tmpdir)

    # with broadcast
    shape2 = (1, 3, 1, 1)
    run_div_test(shape1, shape2, tmpdir)

#Dropout
@pytest.mark.parametrize("dtype", DType_Config)
def test_Dropout(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.asarray([[10, 20],[30, 40],[50, 60]], dtype=dtype)
        model = C.dropout(data, 0.5)
        verify_no_input(model, tmpdir, 'Dropout_0')

        x = C.input_variable(data.shape)
        model = C.dropout(x, 0.5)
        verify_one_input(model, data, tmpdir, 'Dropout_1')

#Elu
@pytest.mark.parametrize("dtype", DType_Config)
def test_Elu(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.asarray([[-1, -0.5, 0, 1, 2]], dtype=dtype)
        model = C.elu(data)
        verify_no_input(model, tmpdir, 'Elu_0')

        x1 = C.input_variable(data.shape)
        model = C.elu(x1)
        verify_one_input(model, data, tmpdir, 'Elu_1')

        x2 = C.input_variable(data.shape)
        model = C.elu(x2, alpha=2.0)
        verify_one_input(model, data, tmpdir, 'Elu_2')

#Equal
@pytest.mark.parametrize("dtype", DType_Config)
def test_Equal(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data0 = np.asarray([41., 42., 43.], dtype=dtype)
        data1 = np.asarray([42., 42., 42.], dtype=dtype)
        model = C.equal(data0, data1)
        verify_no_input(model, tmpdir, 'Equal_0')

#Exp
@pytest.mark.parametrize("dtype", DType_Config)
def test_Exp(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.asarray([0., 1.], dtype=dtype)
        model = C.exp(data)
        verify_no_input(model, tmpdir, 'Exp_0')

        x = C.input_variable(data.shape)
        model = C.exp(x)
        verify_one_input(model, data, tmpdir, 'Exp_1')

#Flatten
@pytest.mark.parametrize("dtype", DType_Config)
def test_Flatten(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        shape = (2, 3, 4, 5)
        data = np.reshape(np.arange(np.prod(shape), dtype = dtype), shape)
        model = C.flatten(data, 1)
        verify_no_input(model, tmpdir, 'Flatten_0')

        x = C.input_variable(data.shape)
        model = C.flatten(x, 1)
        verify_one_input(model, data, tmpdir, 'Flatten_1')

#Floor
@pytest.mark.parametrize("dtype", DType_Config)
def test_Floor(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.asarray([0.2, 1.3, 4., 5.5, 0.0], dtype=dtype)
        model = C.floor(data)
        verify_no_input(model, tmpdir, 'Floor_0')

        x = C.input_variable(data.shape)
        model = C.floor(x)
        verify_one_input(model, data, tmpdir, 'Floor_1')

#Gather
@pytest.mark.parametrize("dtype", DType_Config)
def test_Gather(tmpdir, dtype):
    if (dtype == np.float16):
        pytest.skip("TO BE FIXED")
    with C.default_options(dtype = dtype):
        c = np.asarray([[[0],[1]],[[4],[5]]]).astype(dtype)
        x = C.input_variable((2,1))
        d = np.arange(12).reshape(6,2).astype(dtype)
        y = C.constant(d)
        model = C.gather(y, x)
        verify_one_input(model, c, tmpdir, 'Gather_1')

#Gather
@pytest.mark.parametrize("dtype", DType_Config)
def test_Gather_With_Axis(tmpdir, dtype):
    if (dtype == np.float16):
        pytest.skip("TO BE FIXED")
    with C.default_options(dtype = dtype):
        data = np.asarray( [[ [111, 112], [121, 122], [131, 132], ],[ [211, 212], [221, 222], [231, 232], ]]).astype(dtype)
        indices = np.asarray([[0, 1, 1], [1, 1, 1]])
        x = C.input_variable(np.shape(data))
        y = C.input_variable(np.shape(indices))
        axis = 1
        model = C.gather(data, y, axis)
        verify_one_input(model, indices, tmpdir, 'Gather_With_Axis_1')

#Greater
@pytest.mark.parametrize("dtype", DType_Config)
def test_Greater(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        model = C.greater([41., 42., 43.], [42., 42., 42.])
        verify_no_input(model, tmpdir, 'Greater_0')

#GRU
@pytest.mark.parametrize("dtype", DType_Config)
def test_GRU(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        def MakeGRUNameFromConfig(backward, initial_state, activition):
            model_name = 'GRU.' + activition.__name__
            if (initial_state != 0):
                model_name += '.initial'
            if (backward):
                model_name += '.backward'
            else:    
                model_name += '.forward'
            return model_name 

        direction_options = [False, True]
        activation_options = [C.tanh]
        initial_state_options = [0]

        input_dim = 2
        cell_dim = 3
        batch_size = 1
        sequence_len = 5

        for config in list(product(direction_options, initial_state_options, activation_options)):
            model_filename = MakeGRUNameFromConfig(*config)
            print(model_filename)
            backward, initial_state, activation =  config
    
            x = C.input_variable(input_dim, dynamic_axes=[C.Axis.default_batch_axis(), C.Axis('sequenceAxis')]) 
            GRUModel = C.layers.Recurrence(C.layers.GRU(cell_dim,     
                                                        activation = activation),   
                                           initial_state = initial_state,    
                                           go_backwards=backward)(x)
            data = np.random.uniform(low=0.0, high=1.0, size=(batch_size, sequence_len, input_dim)).astype('f')
            verify_one_input(GRUModel, data, tmpdir, model_filename)


#Hardmax
@pytest.mark.parametrize("dtype", DType_Config)
def test_Hardmax(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.asarray([1., 1., 2., 3.], dtype=dtype)
        model = C.hardmax(data)
        verify_no_input(model, tmpdir, 'Hardmax_0')

#HardSigmiod
@pytest.mark.parametrize("dtype", DType_Config)
def test_HardSigmiod(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        shape = (2,3)
        x = C.input_variable(shape=shape, dtype=dtype)
        alpha = 1.2
        beta = 2.5
        model = C.hard_sigmoid(x, alpha, beta, 'hardSigmoid')

        data = np.random.rand(*shape).astype(dtype)
        verify_one_input(model, data, tmpdir, 'HardSigmoid_1')

#ImageScaler
@pytest.mark.parametrize("dtype", DType_Config)
def test_ImageScaler(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        input_height = 32
        input_width = 32
        channels = 3
        image = np.ones([channels, input_height, input_width]).astype(dtype)
        scalar = 1.5
        bias = [10, 20, 30]

        model = C.image_scaler(image, scalar, bias);
        verify_no_input(model, tmpdir, 'ImageScaler_0')

        x = C.input_variable(np.shape(image)) 
        model = C.image_scaler(x, scalar, bias);
        verify_one_input(model, image, tmpdir, 'ImageScaler_1')

#LayerNormalization
@pytest.mark.parametrize("dtype", DType_Config)
def test_LayerNormalization(tmpdir, dtype):
    if (dtype == np.float16):
        pytest.skip("TO BE FIXED")

    # This test point tests the LayerNormalization round trip with defaultepsilon. We loose always the epsilon value when 
    # exporting to ONNX (because ONNX MeanVarianceNormalization does not have an epsilon attribute). When loading back 
    # from ONNX, CNTK always uses the default eposilon value (0.00001). That's why test below has the default epsilon 
    # value. It is not expected to pass with any other epsilon value until something changes.
    with C.default_options(dtype = dtype):
        test_shapes = [(3, 5, 7), (10, ), (20, 31)]
        for shape in test_shapes:
            data = np.reshape(np.arange(np.prod(shape), dtype = dtype), shape)
            input_operand = C.input_variable(shape=shape)        
            model0 = C.layers.LayerNormalization(initial_scale=1, initial_bias=2, epsilon=0.00001)(input_operand)
            verify_one_input(model0, data, tmpdir, 'LayerNorm_0')

        # This test point tests especially with epsilon = 0, because that creates a graph with 
        # different number of ops. However, we don't expect the numbers to match in round trip
        # because we only support default epislon (0.00001) when loading from ONNX. Therefore,
        # this is just a load/save test.
        model1 = C.layers.LayerNormalization(epsilon=0.0)(input_operand)
        filename = os.path.join(str(tmpdir), R'LayerNorm_1.onnx')
        model1.save(filename, format=C.ModelFormat.ONNX)
        loaded_model = C.Function.load(filename, format=C.ModelFormat.ONNX)
        assert model1.shape == loaded_model.shape

#LeakyRelu
@pytest.mark.parametrize("dtype", DType_Config)
def test_LeakyRelu(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.asarray([[-1, -0.5, 0, 1, 2]], dtype=dtype)
        model = C.leaky_relu(data)
        verify_no_input(model, tmpdir, 'LeakyRelu_0')

#Less
@pytest.mark.parametrize("dtype", DType_Config)
def test_Less(tmpdir, dtype):
    if (dtype == np.float16):
        pytest.skip("TO BE FIXED")

    with C.default_options(dtype = dtype):
        data0 = np.asarray([41., 42., 43.], dtype=dtype)
        data1 = np.asarray([42., 42., 42.], dtype=dtype)

        model = C.less(data0, data1)
        verify_no_input(model, tmpdir, 'Less_0')

#Log
@pytest.mark.parametrize("dtype", DType_Config)
def test_Log(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.asarray([1., 2.], dtype=dtype)
        model = C.log(data)
        verify_no_input(model, tmpdir, 'Log_0')

#LogSoftmax
@pytest.mark.parametrize("dtype", DType_Config)
def test_LogSoftmax(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        model = C.log_softmax(np.array([[1, 1, 2, 3]]).astype(dtype))
        verify_no_input(model, tmpdir, 'LogSoftmax_0')


#LRN
@pytest.mark.parametrize("dtype", DType_Config)
def test_LRN(tmpdir, dtype, device_id):
    if device_id == -1 and dtype == np.float16:
        pytest.skip('Test is skipped on CPU with float16 data')
    device = cntk_device(device_id)
    with C.default_options(dtype=dtype):
        img_shape = (64, 32, 32)
        img = np.asarray(np.random.uniform(-1, 1, img_shape), dtype=dtype)
        x_r = C.input_variable(shape=img_shape, dtype=dtype)
        model = C.local_response_normalization(x_r, 2, 1.0, 0.0001, 0.75)
        verify_one_input(model, img, tmpdir, 'LRN_1', device)

#LSTM
@pytest.mark.parametrize("dtype", DType_Config)
def test_LSTM(tmpdir, dtype):

    with C.default_options(dtype = dtype):
        def CreateLSTMModel(activation, 
                            peepholes, 
                            self_stabilization, 
                            cell_dim, 
                            initial_state):  
            return C.layers.Sequential([  
                C.layers.Recurrence(C.layers.LSTM(cell_dim,  
                                                  use_peepholes = peepholes,  
                                                  activation = activation,     
                                                  enable_self_stabilization = self_stabilization),     
                                    initial_state = initial_state) 
                ])


        def MakeLSTMNameFromConfig(use_peepholes, enable_self_stabilization, initial_state, activition):
            model_name = 'LSTM.' + activition.__name__
            if (use_peepholes):    
                model_name += '.peephole'
            if(enable_self_stabilization):        
                model_name += '.stabilize'
            if (initial_state != 0):
                model_name += '.initial'
            return model_name 

        # lstm attributes
        use_peepholes_options = [False]
        enable_self_stabilization_options = [False]
        activation_options = [C.tanh]

        #Recurrence attributes
        initial_state_options = [0, 0.23]

        input_dim = 2
        cell_dim = 3
        batch_size = 1
        sequence_len = 5

        for config in list(product(use_peepholes_options, enable_self_stabilization_options, 
                                   initial_state_options, activation_options)):
            model_filename = MakeLSTMNameFromConfig(*config)
            use_peepholes, enable_self_stabilization, initial_state, activation =  config
    
            x = C.input_variable(input_dim, dynamic_axes=[C.Axis.default_batch_axis(), C.Axis('sequenceAxis')]) 
            LSTMmodel = CreateLSTMModel(peepholes = use_peepholes,   
                                        activation = activation,
                                        initial_state = initial_state,
                                        cell_dim = cell_dim,
                                        self_stabilization = enable_self_stabilization)(x)
            data = np.random.uniform(low=0.0, high=1.0, size=(batch_size, sequence_len, input_dim)).astype('f')
            verify_one_input(LSTMmodel, data, tmpdir, model_filename)

#MatMul
@pytest.mark.parametrize("dtype", DType_Config)
def test_MatMul(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data0 = np.asarray([[1,2],[3,4]], dtype=dtype)
        data1 = np.asarray([[5],[6]], dtype=dtype)
        model = C.times(data0, data1)
        verify_no_input(model, tmpdir, 'MatMul_0')

#Max
@pytest.mark.parametrize("dtype", DType_Config)
def test_Max(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data0 = np.asarray([1., 1., 1., 1.], dtype=dtype)
        data1 = np.asarray([0.5, 0.25, 0.125, 0.], dtype=dtype)
        model = C.element_max(data0, data1)
        verify_no_input(model, tmpdir, 'Max_0')

#MaxPool
@pytest.mark.parametrize("dtype", DType_Config)
def test_MaxPool(tmpdir, dtype, device_id):
    if device_id == -1 and dtype == np.float16:
        pytest.skip('Test is skipped on CPU with float16 data')
    device = cntk_device(device_id)
    with C.default_options(dtype=dtype):
        img = np.reshape(np.arange(16, dtype = dtype), [1, 4, 4])
        x = C.input_variable(img.shape)
        model = C.pooling(x, C.MAX_POOLING, (2,2), (3,3))
        verify_one_input(model, img, tmpdir, 'MaxPool_1', device)

#MaxRoiPool
@pytest.mark.parametrize("dtype", DType_Config)
def test_MaxRoiPool(tmpdir, dtype):
    pytest.skip('MaxRoiPool is failing with ONNX shape inference (input rois). RuntimeError: [ShapeInferenceError] RoIs tensor must have 2 dimensions')
    with C.default_options(dtype = dtype):
        input_map = [[[1., 2., 3.],       # (1, 3, 3) input operand (conv feature map)
               [4., 5., 6.],
               [7., 8., 9.]]]
        input_rois = [[1, 1, 2, 2]]

        conv_input = np.asarray(input_map, dtype=dtype)
        roi_input = np.asarray(input_rois, dtype=dtype)

        a = C.input_variable(shape=conv_input.shape,
                    dtype=dtype,
                    needs_gradient=True,
                    name='a')

        b = C.input_variable(shape=roi_input.shape,
                    dtype=dtype,
                    needs_gradient=False,
                    name='b')

        # adding batch and sequence axis
        conv_input.shape     = (1,) + conv_input.shape
        roi_input.shape      = (1,) + roi_input.shape

        model = C.roipooling(a, b, C.MAX_POOLING, (3,3), 1.)

        verify_two_input(model, conv_input, roi_input, tmpdir, 'MaxRoiPool_1')

#Mean
@pytest.mark.parametrize("dtype", DType_Config)
def test_Mean(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        in1 = C.input_variable((4,))
        in2 = C.input_variable((4,))
        model = C.mean([in1, in2])

        in1_data = np.asarray([[1., 2., 3., 4.]], dtype = dtype)
        in2_data = np.asarray([[0., 5., -3., 2.]], dtype = dtype)

        verify_two_input(model, in1_data, in2_data, tmpdir, 'Mean_2')
    
#MeanVarianceNormalization
@pytest.mark.parametrize("dtype", DType_Config)
def test_MeanVarianceNormalization(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        shape = (3, 5, 7)
        data = np.reshape(np.arange(np.prod(shape), dtype = dtype), shape)

        input_operand = C.input_variable(shape=shape)

        model0 = C.mean_variance_normalization(input_operand, use_stats_across_channels=False, do_variance_scaling=True)
        verify_one_input(model0, data, tmpdir, 'MVN_0')

        model1 = C.mean_variance_normalization(input_operand, use_stats_across_channels=False, do_variance_scaling=False)
        verify_one_input(model1, data, tmpdir, 'MVN_1')

        model2 = C.mean_variance_normalization(input_operand, use_stats_across_channels=True, do_variance_scaling=True)
        verify_one_input(model2, data, tmpdir, 'MVN_2')

        # The test below tests the round trip with epsilon. We loose always the epsilon value when exporting to ONNX
        # (because ONNX MeanVarianceNormalization does not have an epsilon attribute). When loading back from ONNX, CNTK
        # always uses the default eposilon value (0.00001). That's why test below has the default epsilon value. It is 
        # not expected to pass with any other epsilon value until something changes.
        model3 = C.mean_variance_normalization(input_operand, epsilon=0.00001, use_stats_across_channels=False, do_variance_scaling=True) 
        verify_one_input(model3, data, tmpdir, 'MVN_3')

#Min
@pytest.mark.parametrize("dtype", DType_Config)
def test_Min(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data0 = np.asarray([1., 1., 1., 1.], dtype=dtype)
        data1 = np.asarray([0.5, 0.25, 0.125, 0.], dtype=dtype)
        model = C.element_min(data0, data1)
        verify_no_input(model, tmpdir, 'Min_0')

#Mul
@pytest.mark.parametrize("dtype", DType_Config)
def test_Mul(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data0 = np.asarray([1., 1., 1., 1.], dtype=dtype)
        data1 = np.asarray([0.5, 0.25, 0.125, 0.], dtype=dtype)
        model = C.element_times(data0, data1)
        verify_no_input(model, tmpdir, 'ElementTimes_0')

#Neg
@pytest.mark.parametrize("dtype", DType_Config)
def test_Neg(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data0 = np.asarray([1., -1., -2., 1.], dtype=dtype)
        model = C.negate(data0)
        verify_no_input(model, tmpdir, 'Neg_0')

#OptimizedRNNStack
OPTIM_RNN_STACK_CONFIGS = ((True, 1, 2, 3, 'lstm'), (False, 1, 4, 8, 'lstm'),
                           (True, 2, 2, 3, 'lstm'), (True, 2, 4, 8, 'lstm'), (True, 2, 6, 8, 'lstm'), 
                           (True, 4, 2, 3, 'lstm'), (False, 2, 2, 3, 'lstm'), (False, 2, 6, 8, 'lstm'), (False, 4, 4, 8, 'lstm'),
                           (True, 1, 2, 3, 'rnnReLU'), (True, 4, 4, 8, 'rnnReLU'), (False, 2, 6, 8, 'rnnReLU'), 
                           (True, 4, 2, 3, 'rnnTanh'), (False, 2, 2, 3, 'rnnTanh'), (True, 1, 2, 3, 'rnnTanh'))
@pytest.mark.parametrize("bidirectional, num_layers, input_size, hidden_size, recurrent_op", OPTIM_RNN_STACK_CONFIGS)
def test_OptimizedRNNStack(bidirectional, num_layers, input_size, hidden_size, recurrent_op, tmpdir, device_id):
    pytest.skip('Need to support new ONNX spec.')
    if device_id == -1:
        pytest.skip('Test only runs on GPU')
    dev = cntk_device(device_id)    
    from _cntk_py import constant_initializer
    model_filename = 'optimized_rnn_stack_' + ('bi' if bidirectional else 'uni') + '_layers' + str(num_layers) + '_inp' + str(input_size) + '_hid' + str(hidden_size)
    W = C.parameter((C.InferredDimension, input_size), constant_initializer(0.1), device=dev)
    x = C.sequence.input_variable(shape=(input_size,))
    s = np.asarray(np.random.uniform(-1, 1, (5,input_size)), dtype=np.float32)
    f = C.optimized_rnnstack(x, W, hidden_size, num_layers, bidirectional=bidirectional, recurrent_op=recurrent_op, name='MyRnnStack')
    f.parameters[0].value = np.reshape(np.arange(np.prod(f.parameters[0].value.shape), dtype=np.float32), f.parameters[0].value.shape)
    verify_one_input(f, s, tmpdir, model_filename)

#Pad
@pytest.mark.parametrize("dtype", DType_Config)
def test_Pad(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        shape = (4, 5)
        data = np.random.rand(*shape).astype(dtype)

        model = C.pad(data, pattern=[(1,1),(2,2)], mode=C.ops.CONSTANT_PAD, constant_value=1)
        verify_no_input(model, tmpdir, 'Pad_0')

        x = C.input_variable(shape)
        model = C.pad(x, pattern=[(1,1),(2,2)], mode=C.ops.REFLECT_PAD)

        verify_one_input(model, data, tmpdir, 'Pad_1')

#PRelu
#def test_PRelu(tmpdir):
#    data = np.asarray([[-1, -0.5, 0, 1, 2]])
#    alpha = C.constant(value=[[0.5, 0.5, 0.5, 0.5, 0.5]])
#    model = C.param_relu(alpha, data)
#    verify_no_input(model, tmpdir, 'PRelu_0')

#Pow
@pytest.mark.parametrize("dtype", DType_Config)
def test_Pow(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        model = C.pow(np.array([1, 2, -2]).astype(dtype), np.array([3, -2, 3]).astype(dtype))
        verify_no_input(model, tmpdir, 'Pow_0')

#Reciprocal
@pytest.mark.parametrize("dtype", DType_Config)
def test_Reciprocal(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        model = C.reciprocal(np.array([-1/3, 1/5, -2, 3]).astype(dtype))
        verify_no_input(model, tmpdir, 'Reciprocal_0')

@pytest.mark.parametrize("dtype", DType_Config)
def test_ReduceL1(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.array([[[1,2], [3,4]],[[5,6], [7,8]],[[9,10], [11,12]]], dtype=dtype)
        model = C.reduce_l1(data, 1)
        verify_no_input(model, tmpdir, 'ReduceL1_0')

        x = C.input_variable(np.shape(data))
        model = C.reduce_l1(x, 1)
        verify_one_input(model, data, tmpdir, 'ReduceL1_1')

@pytest.mark.parametrize("dtype", DType_Config)
def test_ReduceL2(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.array([[[1,2], [3,4]],[[5,6], [7,8]],[[9,10], [11,12]]], dtype=dtype)
        model = C.reduce_l2(data, 0)
        verify_no_input(model, tmpdir, 'ReduceL2_0')

@pytest.mark.parametrize("dtype", DType_Config)
def test_ReduceSumSquare(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.array([[[1,2], [3,4]],[[5,6], [7,8]],[[9,10], [11,12]]], dtype=dtype)
        model = C.reduce_sum_square(data, 0)
        verify_no_input(model, tmpdir, 'ReduceSumSquare_0')

#ReduceLogSum
@pytest.mark.parametrize("dtype", DType_Config)
def test_ReduceLogSum(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=dtype)
        model = C.reduce_log_sum_exp(data, axis=0)

    verify_no_input(model, tmpdir, 'ReduceLogSum_0')

#ReduceMax
@pytest.mark.parametrize("dtype", DType_Config)
def test_ReduceMax(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=dtype)
        model = C.reduce_max(data, 0)
        verify_no_input(model, tmpdir, 'ReduceMax_0')

#ReduceMean
@pytest.mark.parametrize("dtype", DType_Config)
def test_ReduceMean(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=dtype)
        model = C.reduce_mean(data, 0)
        verify_no_input(model, tmpdir, 'ReduceMean_0')

#ReduceMin
@pytest.mark.parametrize("dtype", DType_Config)
def test_ReduceMin(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=dtype)
        model = C.reduce_min(data, 0)
        verify_no_input(model, tmpdir, 'ReduceMin_0')

#ReduceProd
@pytest.mark.parametrize("dtype", DType_Config)
def test_ReduceProd(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=dtype)
        model = C.reduce_prod(data, 0)
        verify_no_input(model, tmpdir, 'ReduceProd_0')

#ReduceSum
@pytest.mark.parametrize("dtype", DType_Config)
def test_ReduceSum(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=dtype)
        model = C.reduce_sum(data, 0)
        verify_no_input(model, tmpdir, 'ReduceSum_0')

#Relu
@pytest.mark.parametrize("dtype", DType_Config)
def test_Relu(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.array([[-1, -0.5, 0, 1, 2]], dtype = dtype)
        model = C.relu(data)
        verify_no_input(model, tmpdir, 'Relu_0')

#Reshape
@pytest.mark.parametrize("dtype", DType_Config)
def test_Reshape(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.asarray([[[[0., 1.],[2., 3.],[4., 5.]]]], dtype=dtype)
        i1 = C.input_variable(shape=(3,2))
        model = C.reshape(i1, (2,3))
        verify_one_input(model, data, tmpdir, 'Reshape_1')

#RNN
@pytest.mark.parametrize("dtype", DType_Config)
def test_RNN(tmpdir, dtype):

    with C.default_options(dtype = dtype):
        def CreatRNN(cell_dim, 
                     activation, 
                     initial_state,
                     direction, 
                     num_layers, 
                     init=C.default_override_or(C.glorot_uniform()), 
                     init_bias=C.default_override_or(0)):
            if direction == 'bidirectional':  
                return C.layers.Sequential([  
                    C.layers.For(range(num_layers), lambda i: [  
                        (C.layers.Recurrence(C.layers.RNNStep(cell_dim, 
                                                              activation = activation,    
                                                              init = init,   
                                                              init_bias = init_bias),  
                                    initial_state = initial_state,  
                                    return_full_state = False, go_backwards=False),   
                         C.layers.Recurrence(C.layers.RNNStep(cell_dim, activation = activation,   
                                        init = init,  
                                        init_bias = init_bias), 
                                    initial_state = initial_state,  
                                    return_full_state = False, go_backwards=True)),   
                        C.splice])])
            else:
                go_backward = False if direction == 'forward' else True
                return C.layers.Sequential([ 
                    C.layers.For(range(num_layers), lambda i: [ 
                        C.layers.Recurrence(C.layers.RNNStep(cell_dim, 
                                                             activation = activation,   
                                        init = init,  
                                        init_bias = init_bias),  
                                    initial_state = initial_state,  
                                    return_full_state = False, go_backwards=go_backward)])])

        def MakeRNNNameFromConfig(direction, num_layers, initial_state, activition):
            model_name = 'RNN.' + direction + '.'

            if num_layers == 1:
                model_name += 'one_layer.'
            else:
                assert (num_layers == 2), "needs 1 or 2 layers!"
                model_name += 'two_layer.'

            if (initial_state != 0):
                model_name += 'initial.'
        
            model_name += activition.__name__
            return model_name 

        direction_options = ['forward', 'reverse', 'bidirectional']
        num_layers_options = [1, 2]
        initial_state_options = [0]
        activation_options = [C.tanh, C.relu, C.sigmoid]

        input_dim = 2
        hidden_dim = 3
        batch_size = 1
        sequence_len = 5

        for config in list(product(direction_options, num_layers_options, initial_state_options, activation_options)):
            model_filename = MakeRNNNameFromConfig(*config)
            print(model_filename)
            direction, num_layers, initial_state, activation = config
    
            x = C.input_variable(input_dim, dynamic_axes=[C.Axis.default_batch_axis(), C.Axis('sequenceAxis')]) 
            RNNModel = CreatRNN(
                hidden_dim, 
                activation,  
                initial_state, 
                direction, 
                num_layers)(x)
            data = np.random.uniform(low=0.0, high=1.0, size=(batch_size, sequence_len, input_dim)).astype(dtype)
            verify_one_input(RNNModel, data, tmpdir, model_filename)

#Selu
@pytest.mark.parametrize("dtype", DType_Config)
def test_Selu(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        model = C.selu(np.array([[-1, -0.5, 0, 1, 2]]).astype(dtype))
        verify_no_input(model, tmpdir, 'Selu_0')

#Sigmoid
@pytest.mark.parametrize("dtype", DType_Config)
def test_Sigmoid(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        model = C.sigmoid(np.array([-2, -1., 0., 1., 2.]).astype(dtype))
        verify_no_input(model, tmpdir, 'Sigmoid_0')

#Slice
@pytest.mark.parametrize("dtype", DType_Config)
def test_Slice(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.asarray([[1,2,-3], [4, 5, 6]],dtype=dtype)
        x1 = C.input_variable((2,3))

        model = C.slice(data, 0, 1, 2)
        verify_no_input(model, tmpdir, 'Slice_0')

        model = C.slice(x1, 0, 1, 2)
        verify_one_input(model, data, tmpdir, 'Slice_1')

        model = C.slice(x1, [0,1], [1,0], [2,1]);
        verify_one_input(model, data, tmpdir, 'Slice2_1')

#Softmax
@pytest.mark.parametrize("dtype", DType_Config)
def test_Softmax(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        model = C.softmax(np.array([[1, 1, 2, 3]]).astype(dtype))
        verify_no_input(model, tmpdir, 'Softmax_0')

#Softplus
@pytest.mark.parametrize("dtype", DType_Config)
def test_Softplus(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        model = C.softplus([[-1, -0.5, 0, 1, 2]])
        verify_no_input(model, tmpdir, 'Softplus_0')

#Softsign
@pytest.mark.parametrize("dtype", DType_Config)
def test_Softsign(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        model = C.softsign(np.array([[-1, -0.5, 0, 1, 2]]).astype(dtype))
        verify_no_input(model, tmpdir, 'Softsign_0')

#Squeeze
#def test_Squeeze(tmpdir):
#    x0 = np.arange(12).reshape((2, 2, 1, 3)).astype('f')
#    x = C.input_variable((2, 1, 3))
#    model = C.squeeze(x)
#    verify_one_input(model, x0, tmpdir, 'Squeeze_0')

#Sum
@pytest.mark.parametrize("dtype", DType_Config)
def test_Sum(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        in1_data = np.asarray([[1., 2., 3., 4.]], dtype = dtype)
        in2_data = np.asarray([[0., 5., -3., 2.]], dtype = dtype)

        in1 = C.input_variable(np.shape(in1_data))
        in2 = C.input_variable(np.shape(in2_data))
        model = C.sum([in1, in2])

        verify_two_input(model, in1_data, in2_data, tmpdir, 'Sum_2')

# SpaceToDepth
@pytest.mark.parametrize("dtype", DType_Config)
def test_SpaceToDepth(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        num_channels = 3
        block_size = 3
        image_shape = (12, 15)
        input_val = np.array(np.reshape(range(num_channels), (num_channels, 1, 1)), dtype=dtype)
        input_val = np.tile(input_val, (1,) + image_shape)
        input_val.shape = (1,) + input_val.shape
        img = C.input_variable((num_channels,) + image_shape, dtype=dtype)
        model = C.space_to_depth(img, block_size)

        verify_one_input(model, input_val, tmpdir, 'SpaceToDepth')

#Sqrt
@pytest.mark.parametrize("dtype", DType_Config)
def test_Sqrt(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        model = C.sqrt(np.array([0., 4.]).astype(dtype))
        verify_no_input(model, tmpdir, 'Sqrt_0')

#Sub
@pytest.mark.parametrize("dtype", DType_Config)
def test_Sub(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        model = C.minus(np.array([1, 2, 3]).astype(dtype), np.array([4, 5, 6]).astype(dtype))
        verify_no_input(model, tmpdir, 'Sub_0')

#Tanh
@pytest.mark.parametrize("dtype", DType_Config)
def test_Tanh(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        model = C.tanh(np.array([[1,2],[3,4]]).astype(dtype))
        verify_no_input(model, tmpdir, 'Tanh_0')

#Transpose
@pytest.mark.parametrize("dtype", DType_Config)
def test_Transpose(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.arange(24).reshape(2,3,4).astype(dtype)
        x = C.input_variable(np.shape(data))

        model = C.transpose(data, perm=(2, 0, 1))
        verify_no_input(model, tmpdir, 'Transpose_0')

        model = C.transpose(x, perm=(2, 0, 1))
        verify_one_input(model, data, tmpdir, 'Transpose_1')

        model = C.transpose(x, perm=(0, 2, 1))
        verify_one_input(model, data, tmpdir, 'Transpose_1_2')

#Transpose
@pytest.mark.parametrize("dtype", DType_Config)
def test_TransposeAxes(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.array([[[0,1],[2,3],[4,5]]]).astype(dtype)
        model = C.swapaxes(data, 1, 2)
        verify_no_input(model, tmpdir, 'TransposeAxes_0')

    # TODO: there is probably a bug in C.swapaxes which does not allow 
    # evaluation of model with data
    #x = C.input_variable(np.shape(data))
    #model = C.swapaxes(x, 1, 2)
    #verify_one_input(model, data, tmpdir, 'TransposeAxes_1')


# Select
@pytest.mark.parametrize("flag, if_true, if_false", (
    ((-100., -1.2, -1.0, -0.5, 0.0, 0.1, 1.0, 100.),
     (1., 2., 3., 4., 5., 6., 7., 8.),
     (11., 12., 13., 14., 15., 16., 17., 18.)),
    (((1, 0, 3), (4, 5, 0)),
     ((1, 2, 3), (4, 5, 6)),
     ((-1, -2, -3), (-4, -5, -6))),
    ((((0, 1), (0, 1)), ((0, 1), (0, 1))),
     (((1, 2), (3, 4)), ((5, 6), (7, 8))),
     (((9, 10), (11, 12)), ((13, 14), (15, 16)))),
))
def test_Select(flag, if_true, if_false, tmpdir):
    flag = np.asarray(flag, dtype=np.float32)
    if_true = np.asarray(if_true, dtype=np.float32)
    if_false = np.asarray(if_false, dtype=np.float32)

    model = C.element_select(flag, if_true, if_false)
    verify_no_input(model, tmpdir, 'Select_0')

    flag_var = C.input_variable(np.shape(flag))
    if_true_var = C.input_variable(np.shape(if_true))
    if_false_var = C.input_variable(np.shape(if_false))

    model = C.element_select(flag_var, if_true, if_false)
    verify_one_input(model, flag, tmpdir, 'Select_1_flag')

    model = C.element_select(flag, if_true_var, if_false)
    verify_one_input(model, if_true, tmpdir, 'Select_1_if_true')

    model = C.element_select(flag, if_true, if_false_var)
    verify_one_input(model, if_false, tmpdir, 'Select_1_if_false')

# Cos
@pytest.mark.parametrize("dtype", DType_Config)
def test_Cos(tmpdir, dtype):
    data = np.asarray([0.0, -0.5, 0.5, 10, 20], dtype)
    model = C.cos(data)
    verify_no_input(model, tmpdir, 'Cos_0')

# Sin
@pytest.mark.parametrize("dtype", DType_Config)
def test_Sin(tmpdir, dtype):
    data = np.asarray([0.0, -0.5, 0.5, 10, 20], dtype)
    model = C.sin(data)
    verify_no_input(model, tmpdir, 'Sin_0')

# Tan
@pytest.mark.parametrize("dtype", DType_Config)
def test_Tan(tmpdir, dtype):
    data = np.asarray([0.0, -0.5, 0.5, 10, 20], dtype)
    model = C.tan(data)
    verify_no_input(model, tmpdir, 'Tan_0')

# Acos
@pytest.mark.parametrize("dtype", DType_Config)
def test_Acos(tmpdir, dtype):
    data = np.asarray([0.0, -0.5, 0.5, 1, -1], dtype)
    model = C.acos(data)
    verify_no_input(model, tmpdir, 'Acos_0')

# Asin
@pytest.mark.parametrize("dtype", DType_Config)
def test_Asin(tmpdir, dtype):
    data = np.asarray([0.0, -0.5, 0.5, 1, -1], dtype)
    model = C.asin(data)
    verify_no_input(model, tmpdir, 'Asin_0')

# Atan
@pytest.mark.parametrize("dtype", DType_Config)
def test_Atan(tmpdir, dtype):
    data = np.asarray([0.0, -0.5, 0.5, 1, -1], dtype)
    model = C.atan(data)
    verify_no_input(model, tmpdir, 'Atan_0')
