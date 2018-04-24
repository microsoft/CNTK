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

    if (type(o0) is list):
        o0 = o0[0]
    if (type(o1) is list):
        o1 = o1[0]

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

#And
def test_And(tmpdir):
    data1 = np.asarray([[1, 1, 0, 0],[1, 1, 1, 1]], np.float32)
    data2 = np.asarray([1, 0, 1, 0], np.float32)

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
def test_Not(tmpdir):
    data1 = np.asarray([[1, 1, 0, 0],[1, 1, 1, 1]], np.float32)

    model = C.element_not(data1)
    verify_no_input(model, tmpdir, 'Not_0')

    x = C.input_variable(np.shape(data1))

    model = C.element_not(x)
    verify_one_input(model, data1, tmpdir, 'Not_1')

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

    verify_one_input(model, img, tmpdir, 'AveragePool_1')

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

    verify_one_input(model, data1, tmpdir, 'Concat_1')

def test_ConvTranspose(tmpdir):
    # Keep the shapes below as they are, because this tests an earlier bug.
    input_shape = (48, 16, 16) 
    img = np.reshape(np.arange(np.prod(input_shape), dtype = np.float32), input_shape) 

    x = C.input_variable(input_shape)

    kernel_shape = (48, 32, 3, 3) # For convolution_transpose the shape is (I x O x W x H)
    kernel = C.constant(value = np.ones(shape=(kernel_shape), dtype = np.float32))

    conv_trans_model = C.convolution_transpose(kernel, x, strides=(2, 2), output_shape=(32, 32, 32), auto_padding = [False, True, True])

    verify_one_input(conv_trans_model, img, tmpdir, 'ConvTranspose_0')

# DepthToSpace
def test_DepthToSpace(tmpdir):
    num_channels = 9
    block_size = 3
    image_shape = (4, 5)
    input_val = np.array(np.reshape(range(num_channels), (num_channels, 1, 1)), dtype=np.float32)
    input_val = np.tile(input_val, (1,) + image_shape)
    input_val.shape = (1,) + input_val.shape
    img = C.input_variable((num_channels,) + image_shape, dtype=np.float32)
    model = C.depth_to_space(img, block_size)

    verify_one_input(model, input_val, tmpdir, 'DepthToSpace')

#Div
def test_Div(tmpdir):
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

#Flatten
def test_Flatten(tmpdir):
    shape = (2, 3, 4, 5)
    data = np.reshape(np.arange(np.prod(shape), dtype = np.float32), shape)
    model = C.flatten(data, 1)
    verify_no_input(model, tmpdir, 'Flatten_0')

    x = C.input_variable(data.shape)
    model = C.flatten(x, 1)
    verify_one_input(model, data, tmpdir, 'Flatten_1')

#Floor
def test_Floor(tmpdir):
    data = np.asarray([0.2, 1.3, 4., 5.5, 0.0], dtype=np.float32)
    model = C.floor(data)
    verify_no_input(model, tmpdir, 'Floor_0')

    x = C.input_variable(data.shape)
    model = C.floor(x)
    verify_one_input(model, data, tmpdir, 'Floor_1')

#Gather
def test_Gather(tmpdir):
    c = np.asarray([[[0],[1]],[[4],[5]]]).astype('f')
    x = C.input_variable((2,1))
    d = np.arange(12).reshape(6,2).astype('f')
    y = C.constant(d)
    model = C.gather(y, x)
    verify_one_input(model, c, tmpdir, 'Gather_1')

#Gather
def test_Gather_With_Axis(tmpdir):
    data = np.asarray( [[ [111, 112], [121, 122], [131, 132], ],[ [211, 212], [221, 222], [231, 232], ]]).astype('f')
    indices = np.asarray( [ [0, 1, 1], [1, 1, 1]])
    x = C.input_variable(np.shape(data))
    y = C.input_variable(np.shape(indices))
    axis = 1
    model = C.gather(data, y, axis)
    verify_one_input(model, indices, tmpdir, 'Gather_With_Axis_1')

#Greater
def test_Greater(tmpdir):
    model = C.greater([41., 42., 43.], [42., 42., 42.])
    verify_no_input(model, tmpdir, 'Greater_0')

#GRU
def test_GRU(tmpdir):
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
        #CLG.plot(GRUModel, filename=cntk_pdf_filename)
        #plot_block_internals(GRUModel, 'GRU', model_filename)
        data = np.random.uniform(low=0.0, high=1.0, size=(batch_size, sequence_len, input_dim)).astype('f')
        verify_one_input(GRUModel, data, tmpdir, model_filename)


#Hardmax
def test_Hardmax(tmpdir):
    data = np.asarray([1., 1., 2., 3.], dtype=np.float32)
    model = C.hardmax(data)
    verify_no_input(model, tmpdir, 'Hardmax_0')

#HardSigmiod
def test_HardSigmiod(tmpdir):
    shape = (2,3)
    x = C.input_variable(shape=shape, dtype=np.float32)
    alpha = 1.2
    beta = 2.5
    model = C.hard_sigmoid(x, alpha, beta, 'hardSigmoid')

    data = np.random.rand(*shape).astype(np.float32)
    verify_one_input(model, data, tmpdir, 'HardSigmoid_1')

#ImageScaler
def test_ImageScaler(tmpdir):
    input_height = 32
    input_width = 32
    channels = 3
    image = np.ones([channels, input_height, input_width]).astype(np.float32)
    scalar = 1.5
    bias = [10, 20, 30]

    model = C.image_scaler(image, scalar, bias);
    verify_no_input(model, tmpdir, 'ImageScaler_0')

    x = C.input_variable(np.shape(image)) 
    model = C.image_scaler(x, scalar, bias);
    verify_one_input(model, image, tmpdir, 'ImageScaler_1')

#LayerNormalization
def test_LayerNormalization(tmpdir):
    # This test point tests the LayerNormalization round trip with defaultepsilon. We loose always the epsilon value when 
    # exporting to ONNX (because ONNX MeanVarianceNormalization does not have an epsilon attribute). When loading back 
    # from ONNX, CNTK always uses the default eposilon value (0.00001). That's why test below has the default epsilon 
    # value. It is not expected to pass with any other epsilon value until something changes.
    test_shapes = [(3, 5, 7), (10, ), (20, 31)]
    for shape in test_shapes:
        data = np.reshape(np.arange(np.prod(shape), dtype = np.float32), shape)
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

#LogSoftmax
def test_LogSoftmax(tmpdir):
    model = C.log_softmax([[1, 1, 2, 3]])
    verify_no_input(model, tmpdir, 'LogSoftmax_0')


#LRN
def test_LRN(tmpdir):
    img_shape = (64, 32, 32)
    img = np.asarray(np.random.uniform(-1, 1, img_shape), dtype=np.float32)
    x_r = C.input_variable(shape=img_shape, dtype=np.float32)
    model = C.local_response_normalization(x_r, 2, 1.0, 0.0001, 0.75)
    verify_one_input(model, img, tmpdir, 'LRN_1')

#LSTM
def test_LSTM(tmpdir):
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
    initial_state_options = [0]

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

#Mean
def test_Mean(tmpdir):
    in1 = C.input_variable((4,))
    in2 = C.input_variable((4,))
    model = C.mean([in1, in2])

    in1_data = np.asarray([[1., 2., 3., 4.]], np.float32)
    in2_data = np.asarray([[0., 5., -3., 2.]], np.float32)

    verify_two_input(model, in1_data, in2_data, tmpdir, 'Mean_2')
    
#MeanVarianceNormalization
def test_MeanVarianceNormalization(tmpdir):
    shape = (3, 5, 7)
    data = np.reshape(np.arange(np.prod(shape), dtype = np.float32), shape)

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

#OptimizedRNNStack
OPTIM_RNN_STACK_CONFIGS = ((True, 2, 2, 3, 'lstm'), (True, 2, 4, 8, 'lstm'), (True, 2, 6, 8, 'lstm'), 
                           (True, 4, 2, 3, 'lstm'), (False, 2, 2, 3, 'lstm'),
                           (True, 1, 2, 3, 'rnnReLU'), (True, 4, 4, 8, 'rnnReLU'), (False, 2, 6, 8, 'rnnReLU'), 
                           (True, 4, 2, 3, 'rnnTanh'), (False, 2, 2, 3, 'rnnTanh'), (True, 1, 2, 3, 'rnnTanh'))
@pytest.mark.parametrize("bidirectional, num_layers, input_size, hidden_size, recurrent_op", OPTIM_RNN_STACK_CONFIGS)
def test_OptimizedRNNStack(bidirectional, num_layers, input_size, hidden_size, recurrent_op, tmpdir, device_id):
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
def test_Pad(tmpdir):
    shape = (4, 5)
    data = np.random.rand(*shape).astype(np.float32)

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
def test_Pow(tmpdir):
    model = C.pow([1, 2, -2], [3, -2, 3])
    verify_no_input(model, tmpdir, 'Pow_0')

#Reciprocal
def test_Reciprocal(tmpdir):
    model = C.reciprocal([-1/3, 1/5, -2, 3])
    verify_no_input(model, tmpdir, 'Reciprocal_0')

def test_ReduceL1(tmpdir):
    data = np.array([[[1,2], [3,4]],[[5,6], [7,8]],[[9,10], [11,12]]], dtype=np.float32)
    model = C.reduce_l1(data, 1)
    verify_no_input(model, tmpdir, 'ReduceL1_0')

    x = C.input_variable(np.shape(data))
    model = C.reduce_l1(x, 1)
    verify_one_input(model, data, tmpdir, 'ReduceL1_1')

def test_ReduceL2(tmpdir):
    data = np.array([[[1,2], [3,4]],[[5,6], [7,8]],[[9,10], [11,12]]], dtype=np.float32)
    model = C.reduce_l2(data, 0)
    verify_no_input(model, tmpdir, 'ReduceL2_0')

def test_ReduceSumSquare(tmpdir):
    data = np.array([[[1,2], [3,4]],[[5,6], [7,8]],[[9,10], [11,12]]], dtype=np.float32)
    model = C.reduce_sum_square(data, 0)
    verify_no_input(model, tmpdir, 'ReduceSumSquare_0')

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

#RNN
def test_RNN(tmpdir):
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
        data = np.random.uniform(low=0.0, high=1.0, size=(batch_size, sequence_len, input_dim)).astype('f')
        verify_one_input(RNNModel, data, tmpdir, model_filename)

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
    data = np.asarray([[1,2,-3], [4, 5, 6]],dtype=np.float32)
    x1 = C.input_variable((2,3))

    model = C.slice(data, 0, 1, 2)
    verify_no_input(model, tmpdir, 'Slice_0')

    model = C.slice(x1, 0, 1, 2)
    verify_one_input(model, data, tmpdir, 'Slice_1')

    model = C.slice(x1, [0,1], [1,0], [2,1]);
    verify_one_input(model, data, tmpdir, 'Slice2_1')

#Softmax
def test_Softmax(tmpdir):
    model = C.softmax([[1, 1, 2, 3]])
    verify_no_input(model, tmpdir, 'Softmax_0')

#Softplus
def test_Softplus(tmpdir):
    model = C.softplus([[-1, -0.5, 0, 1, 2]])
    verify_no_input(model, tmpdir, 'Softplus_0')

#Softsign
def test_Softsign(tmpdir):
    model = C.softsign([[-1, -0.5, 0, 1, 2]])
    verify_no_input(model, tmpdir, 'Softsign_0')

#Squeeze
#def test_Squeeze(tmpdir):
#    x0 = np.arange(12).reshape((2, 2, 1, 3)).astype('f')
#    x = C.input_variable((2, 1, 3))
#    model = C.squeeze(x)
#    verify_one_input(model, x0, tmpdir, 'Squeeze_0')

#Sum
def test_Sum(tmpdir):
    in1_data = np.asarray([[1., 2., 3., 4.]], np.float32)
    in2_data = np.asarray([[0., 5., -3., 2.]], np.float32)

    in1 = C.input_variable(np.shape(in1_data))
    in2 = C.input_variable(np.shape(in2_data))
    model = C.sum([in1, in2])

    verify_two_input(model, in1_data, in2_data, tmpdir, 'Sum_2')

# SpaceToDepth
def test_SpaceToDepth(tmpdir):
    num_channels = 3
    block_size = 3
    image_shape = (12, 15)
    input_val = np.array(np.reshape(range(num_channels), (num_channels, 1, 1)), dtype=np.float32)
    input_val = np.tile(input_val, (1,) + image_shape)
    input_val.shape = (1,) + input_val.shape
    img = C.input_variable((num_channels,) + image_shape, dtype=np.float32)
    model = C.space_to_depth(img, block_size)

    verify_one_input(model, input_val, tmpdir, 'SpaceToDepth')

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
    data = np.arange(24).reshape(2,3,4).astype('f')
    x = C.input_variable(np.shape(data))

    model = C.transpose(data, perm=(2, 0, 1))
    verify_no_input(model, tmpdir, 'Transpose_0')

    model = C.transpose(x, perm=(2, 0, 1))
    verify_one_input(model, data, tmpdir, 'Transpose_1')

    model = C.transpose(x, perm=(0, 2, 1))
    verify_one_input(model, data, tmpdir, 'Transpose_1_2')

#Transpose
def test_TransposeAxes(tmpdir):
    data = [[[0,1],[2,3],[4,5]]]
    model = C.swapaxes(data, 1, 2)
    verify_no_input(model, tmpdir, 'TransposeAxes_0')

    # TODO: there is probably a bug in C.swapaxes which does not allow 
    # evaluation of model with data
    #x = C.input_variable(np.shape(data))
    #model = C.swapaxes(x, 1, 2)
    #verify_one_input(model, data, tmpdir, 'TransposeAxes_1')

