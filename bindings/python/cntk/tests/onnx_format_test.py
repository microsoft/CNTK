# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import numpy as np
import cntk as C
import pytest

def test_load_save_constant(tmpdir):
    c = C.constant(value=[1,3])
    root_node = c * 5

    result = root_node.eval()
    expected = [[[[5,15]]]]
    assert np.allclose(result, expected)

    filename = os.path.join(str(tmpdir), R'c_plus_c.onnx')
    root_node.save(filename, format=C.ModelFormat.ONNX)

    loaded_node = C.Function.load(filename, format=C.ModelFormat.ONNX)
    assert root_node.shape == loaded_node.shape

    loaded_result = loaded_node.eval()
    assert np.allclose(loaded_result, expected)

def test_dense_layer(tmpdir):
    img_shape = (1, 5, 5)
    img = np.asarray(np.random.uniform(-1, 1, img_shape), dtype=np.float32)

    x = C.input_variable(img.shape)
    root_node = C.layers.Dense(5, activation=C.softmax)(x)
    
    filename = os.path.join(str(tmpdir), R'dense_layer.onnx')
    root_node.save(filename, format=C.ModelFormat.ONNX)

    loaded_node = C.Function.load(filename, format=C.ModelFormat.ONNX)
    assert root_node.shape == loaded_node.shape

    x_ = loaded_node.arguments[0]
    assert np.allclose(loaded_node.eval({x_:img}), root_node.eval({x:img}))

CONVOLUTION_TEST_DATA = [
    # auto_padding: Value for the auto_adding parameter to convolution. 
    ([False, True, True]   # Equivalent to "SAME_UPPER".
     ),
    ([False, False, False] # Equivalent to "VALID" padding.     
     ),
    ([False, False, True]  # Equivalent to "VALID" padding.     
     ),
    ([False, True, False]  # Equivalent to "VALID" padding.     
     )
]
# This is a roundtrip test. It saves a CNTK convolution node in ONNX format (with different padding options), 
# and loads it back to check that the same results are produced.
@pytest.mark.parametrize("auto_padding", CONVOLUTION_TEST_DATA)
def test_convolution(tmpdir, auto_padding):
    img_shape = (1, 5, 5)
    img = np.asarray(np.random.uniform(-1, 1, img_shape), dtype=np.float32)

    x = C.input_variable(img.shape)
    filter = np.reshape(np.array([2, -1, -1, 2], dtype = np.float32), (1, 1, 2, 2))
    kernel = C.constant(value = filter)
    root_node = C.convolution(kernel, x, auto_padding=auto_padding)

    filename = os.path.join(str(tmpdir), R'conv.onnx')
    root_node.save(filename, format=C.ModelFormat.ONNX)

    loaded_node = C.Function.load(filename, format=C.ModelFormat.ONNX)
    assert root_node.shape == loaded_node.shape

    x_ = loaded_node.arguments[0]
    assert np.allclose(loaded_node.eval({x_:[img]}), root_node.eval({x:[img]}))

def test_convolution_transpose(tmpdir):
    img_shape = (1, 3, 3)
    img = np.asarray(np.random.uniform(-1, 1, img_shape), dtype=np.float32)

    x = C.input_variable(img.shape)
    filter = np.reshape(np.array([2, -1, -1, 2], dtype = np.float32), (1, 2, 2))
    kernel = C.constant(value = filter)
    root_node = C.convolution_transpose(kernel, x, auto_padding=[False], output_shape=(1, 4, 4))
    
    filename = os.path.join(str(tmpdir), R'conv_transpose.onnx')
    root_node.save(filename, format=C.ModelFormat.ONNX)

    loaded_node = C.Function.load(filename, format=C.ModelFormat.ONNX)
    assert root_node.shape == loaded_node.shape

    x_ = loaded_node.arguments[0]
    assert np.allclose(loaded_node.eval({x_:[img]}), root_node.eval({x:[img]}))

POOLING_TEST_DATA = [
    # auto_padding: Value for the auto_adding parameter to pooling. 
    ([True, True],   # Equivalent to "SAME_UPPER".
    True                    # pooling_type: True := MaxPooling, False := AveragePooling
     ),
    ([False, False], # Equivalent to "VALID" padding.
    True
     ),
    ([False, True],  # Equivalent to "VALID" padding.
    True     
     ),
    ([True, True],   # Equivalent to "SAME_UPPER".
    False
     ),
    ([False, False], # Equivalent to "VALID" padding.
    False
     ),
    # Enable this test case when we fix ONNX AveragePool to match even for edge pixels in {False, True] case.}
    # ([False, True],  # Equivalent to "VALID" padding.
    # False     
    #  )
]
# This is a roundtrip test. It saves a CNTK pooling node in ONNX format (with different padding options), 
# and loads it back to check that the same results are produced.
@pytest.mark.parametrize("auto_padding, pooling_type", POOLING_TEST_DATA)
def test_pooling(tmpdir, auto_padding, pooling_type):
    img_shape = (1, 5, 5)
    img = np.asarray(np.random.uniform(-1, 1, img_shape), dtype=np.float32)

    x = C.input_variable(img.shape)    
    pool_type = C.MAX_POOLING if pooling_type else C.AVG_POOLING
    root_node = C.pooling(x, pool_type, (2, 2), auto_padding=auto_padding)

    filename = os.path.join(str(tmpdir), R'conv.onnx')
    root_node.save(filename, format=C.ModelFormat.ONNX)

    loaded_node = C.Function.load(filename, format=C.ModelFormat.ONNX)
    assert root_node.shape == loaded_node.shape

    x_ = loaded_node.arguments[0]
    assert np.allclose(loaded_node.eval({x_:[img]}), root_node.eval({x:[img]}))

def test_conv_model(tmpdir):
    def create_model(input):
        with C.layers.default_options(init=C.glorot_uniform(), activation=C.relu):
            model = C.layers.Sequential([
                C.layers.For(range(3), lambda i: [
                    C.layers.Convolution((5,5), [32,32,64][i], pad=True),
                    C.layers.MaxPooling((3,3), strides=(2,2))
                    ]),
                C.layers.Dense(64),
                C.layers.Dense(10, activation=None)
            ])

        return model(input)

    img_shape = (3, 32, 32)
    img = np.asarray(np.random.uniform(-1, 1, img_shape), dtype=np.float32)

    x = C.input_variable(img.shape)
    root_node = create_model(x)

    filename = os.path.join(str(tmpdir), R'conv_model.onnx')
    root_node.save(filename, format=C.ModelFormat.ONNX)

    loaded_node = C.Function.load(filename, format=C.ModelFormat.ONNX)
    assert root_node.shape == loaded_node.shape

    x_ = loaded_node.arguments[0]
    assert np.allclose(loaded_node.eval({x_:img}), root_node.eval({x:img}))

def test_batch_norm_model(tmpdir):
    image_height = 32
    image_width  = 32
    num_channels = 3
    num_classes  = 10

    input_var = C.input_variable((num_channels, image_height, image_width))
    label_var = C.input_variable((num_classes))
    def create_basic_model_with_batch_normalization(input, out_dims):
        with C.layers.default_options(activation=C.relu, init=C.glorot_uniform()):
            model = C.layers.Sequential([
                C.layers.For(range(3), lambda i: [
                    C.layers.Convolution((5,5), [image_width,image_height,64][i], pad=True),
                    C.layers.BatchNormalization(map_rank=1),
                    C.layers.MaxPooling((3,3), strides=(2,2))
                ]),
                C.layers.Dense(64),
                C.layers.BatchNormalization(map_rank=1),
                C.layers.Dense(out_dims, activation=None)
            ])

        return model(input)

    feature_scale = 1.0 / 256.0
    input_var_norm = C.element_times(feature_scale, input_var)
    
    # apply model to input
    z = create_basic_model_with_batch_normalization(input_var_norm, out_dims=10)

    filename = os.path.join(str(tmpdir), R'bn_model.onnx')
    z.save(filename, format=C.ModelFormat.ONNX)

    loaded_node = C.Function.load(filename, format=C.ModelFormat.ONNX)
    assert z.shape == loaded_node.shape

    img_shape = (num_channels, image_width, image_height)
    img = np.asarray(np.random.uniform(-1, 1, img_shape), dtype=np.float32)

    x = z.arguments[0];
    x_ = loaded_node.arguments[0]
    assert np.allclose(loaded_node.eval({x_:img}), z.eval({x:img}))

def test_vgg9_model(tmpdir):
    def create_model(input):
        with C.layers.default_options(activation=C.relu, init=C.glorot_uniform()):
            model = C.layers.Sequential([
                C.layers.For(range(3), lambda i: [
                    C.layers.Convolution((3,3), [64,96,128][i], pad=True),
                    C.layers.Convolution((3,3), [64,96,128][i], pad=True),
                    C.layers.MaxPooling((3,3), strides=(2,2))
                ]),
                C.layers.For(range(2), lambda : [
                    C.layers.Dense(1024)
                ]),
                C.layers.Dense(10, activation=None)
            ])
        
        return model(input)

    img_shape = (3, 32, 32)
    img = np.asarray(np.random.uniform(-1, 1, img_shape), dtype=np.float32)

    x = C.input_variable(img.shape)
    root_node = create_model(x)

    filename = os.path.join(str(tmpdir), R'vgg9_model.onnx')
    root_node.save(filename, format=C.ModelFormat.ONNX)

    loaded_node = C.Function.load(filename, format=C.ModelFormat.ONNX)
    assert root_node.shape == loaded_node.shape

    x_ = loaded_node.arguments[0]
    assert np.allclose(loaded_node.eval({x_:img}), root_node.eval({x:img}))

def test_conv3d_model(tmpdir):
    def create_model(input):
        with C.default_options (activation=C.relu):
            model = C.layers.Sequential([
                    C.layers.Convolution3D((3,3,3), 64, pad=True),
                    C.layers.MaxPooling((1,2,2), (1,2,2)),
                    C.layers.For(range(3), lambda i: [
                        C.layers.Convolution3D((3,3,3), [96, 128, 128][i], pad=True),
                        C.layers.Convolution3D((3,3,3), [96, 128, 128][i], pad=True),
                        C.layers.MaxPooling((2,2,2), (2,2,2))
                    ]),
                    C.layers.For(range(2), lambda : [
                        C.layers.Dense(1024), 
                        C.layers.Dropout(0.5)
                    ]),
                C.layers.Dense(100, activation=None)
            ])

        return model(input)

    video_shape = (3, 20, 32, 32)
    video = np.asarray(np.random.uniform(-1, 1, video_shape), dtype=np.float32)

    x = C.input_variable(video.shape)
    root_node = create_model(x)

    filename = os.path.join(str(tmpdir), R'conv3d_model.onnx')
    root_node.save(filename, format=C.ModelFormat.ONNX)

    loaded_node = C.Function.load(filename, format=C.ModelFormat.ONNX)
    assert root_node.shape == loaded_node.shape

    x_ = loaded_node.arguments[0]
    assert np.allclose(loaded_node.eval({x_:video}), root_node.eval({x:video}))

def test_resnet_model(tmpdir):
    def convolution_bn(input, filter_size, num_filters, strides=(1,1), init=C.normal(0.01), activation=C.relu):
        r = C.layers.Convolution(filter_size, 
                                 num_filters, 
                                 strides=strides, 
                                 init=init, 
                                 activation=None, 
                                 pad=True, bias=False)(input)
        r = C.layers.BatchNormalization(map_rank=1)(r)
        r = r if activation is None else activation(r)    
        return r

    def resnet_basic(input, num_filters):
        c1 = convolution_bn(input, (3,3), num_filters)
        c2 = convolution_bn(c1, (3,3), num_filters, activation=None)
        p  = c2 + input
        return C.relu(p)

    def resnet_basic_inc(input, num_filters):
        c1 = convolution_bn(input, (3,3), num_filters, strides=(2,2))
        c2 = convolution_bn(c1, (3,3), num_filters, activation=None)

        s = convolution_bn(input, (1,1), num_filters, strides=(2,2), activation=None)
    
        p = c2 + s
        return C.relu(p)

    def resnet_basic_stack(input, num_filters, num_stack):
        assert (num_stack > 0)
    
        r = input
        for _ in range(num_stack):
            r = resnet_basic(r, num_filters)
        return r

    def create_model(input):
        conv = convolution_bn(input, (3,3), 16)
        r1_1 = resnet_basic_stack(conv, 16, 3)

        r2_1 = resnet_basic_inc(r1_1, 32)
        r2_2 = resnet_basic_stack(r2_1, 32, 2)

        r3_1 = resnet_basic_inc(r2_2, 64)
        r3_2 = resnet_basic_stack(r3_1, 64, 2)

        # Global average pooling
        pool = C.layers.AveragePooling(filter_shape=(8,8), strides=(1,1))(r3_2)    
        return C.layers.Dense(10, init=C.normal(0.01), activation=None)(pool)

    img_shape = (3, 32, 32)
    img = np.asarray(np.random.uniform(0, 1, img_shape), dtype=np.float32)

    x = C.input_variable(img.shape)
    root_node = create_model(x)

    filename = os.path.join(str(tmpdir), R'resnet_model.onnx')
    root_node.save(filename, format=C.ModelFormat.ONNX)

    loaded_node = C.Function.load(filename, format=C.ModelFormat.ONNX)
    assert root_node.shape == loaded_node.shape

    x_ = loaded_node.arguments[0]
    assert np.allclose(loaded_node.eval({x_:img}), root_node.eval({x:img}))