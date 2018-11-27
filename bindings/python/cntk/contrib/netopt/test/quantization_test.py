import numpy as np
import pytest
import cntk as C
import cntk.contrib.netopt.quantization as qc
from cntk.layers import BatchNormalization
C.cntk_py.set_fixed_random_seed(1)

inC, inH, inW = 1, 28, 28
num_classes = 10
feature_var = C.input_variable((inC, inH, inW))
label_var = C.input((num_classes))
dat = np.ones([1, inC, inH, inW], dtype = np.float32)

# create a network with convolutions
def _create_convolution_model():

    with C.layers.default_options(init=C.glorot_uniform(), activation=C.relu):
        h = feature_var
        # The first two layers has bias=False to test, the conversion
        # work with and without bias in the Convolution.
        h = C.layers.Convolution2D(filter_shape=(5,5),
                                   num_filters=64,
                                   strides=(2,2),
                                   pad=True, bias=False, name='first_convo')(h)        

        h = C.layers.Convolution2D(filter_shape=(5,5),
                                   num_filters=64,
                                   strides=(2,2),
                                   pad=True, bias=False, name='second_convo')(h)

        h = C.layers.Convolution2D(filter_shape=(5,5),
                                   num_filters=64,
                                   strides=(1,1),
                                   pad=True, name='thrid_convo')(h)

        h = C.layers.Convolution2D(filter_shape=(5,5),
                                   num_filters=64,
                                   strides=(1,1),
                                   pad=True, name='fourth_convo')(h)

        r = C.layers.Dense(num_classes, activation=None, name='classify')(h)
    return r


# create a network with convolutions for the tests with skip level links
def _create_convolution_model_with_skip_level_links():

    with C.layers.default_options(init=C.glorot_uniform(), activation=C.relu):
        h = feature_var
        # The first two layers has bias=False to test, the conversion
        # work with and without bias in the Convolution.
        a = C.layers.Convolution2D(filter_shape=(5,5),
                                   num_filters=64,
                                   strides=(2,2),
                                   pad=True, bias=False, name='first_convo')(h)

        a = BatchNormalization(map_rank=1, 
                               normalization_time_constant=4096, 
                               use_cntk_engine=True, init_scale=1, 
                               disable_regularization=True)(a)
        
        b = C.layers.Convolution2D(filter_shape=(5,5),
                                   num_filters=64,
                                   strides=(2,2),
                                   pad=True, bias=False, name='second_convo')(h)

        b = BatchNormalization(map_rank=1, 
                               normalization_time_constant=4096, 
                               use_cntk_engine=True, init_scale=1, 
                               disable_regularization=True)(b)

        h = a + b

        h = C.layers.Convolution2D(filter_shape=(5,5),
                                   num_filters=64,
                                   strides=(1,1),
                                   pad=True, name='thrid_convo')(h)

        h = BatchNormalization(map_rank=1, 
                               normalization_time_constant=4096, 
                               use_cntk_engine=True, init_scale=1, 
                               disable_regularization=True)(h)

        h = C.layers.Convolution2D(filter_shape=(5,5),
                                   num_filters=64,
                                   strides=(1,1),
                                   pad=True, name='fourth_convo')(h)

        h = BatchNormalization(map_rank=1, 
                               normalization_time_constant=4096, 
                               use_cntk_engine=True, init_scale=1, 
                               disable_regularization=True)(h)

        r = C.layers.Dense(num_classes, activation=None, name='classify')(h)
    return r


# Exclude the first convolution layer.
def _filter(convolution_block):
    if convolution_block.name and convolution_block.name != 'first_convo':
        return True
    else:
        return False

def test_binarization():
    z = _create_convolution_model_with_skip_level_links()
    binz = qc.convert_to_binary_convolution(z)

    blocks = C.logging.graph.depth_first_search(
                binz, (lambda x : type(x) == C.Function and x.is_block and x.op_name =='BinaryConvolution') , depth = 0)

    assert(len(blocks) == 4) # all convolution blocks should be converted.

    binz = qc.convert_to_binary_convolution(z, _filter)

    blocks = C.logging.graph.depth_first_search(
                binz, (lambda x : type(x) == C.Function and x.is_block and x.op_name =='BinaryConvolution') , depth = 0)

    assert(len(blocks) == 3) # now only three of them should be converted.
    assert(all(b.op_name != 'first_convo' for b in blocks))


def test_native_convolution(tmpdir):  
    # this test needs native binary convolution library built with halide.
    if not C.contrib.netopt.native_convolve_function_registered:     
        pytest.skip()

    z = _create_convolution_model()
    binz = qc.convert_to_binary_convolution(z, _filter)
    
    # save and load to transfer the model to CPU device as native binary
    # convolution does not run on GPU yet.
    model_file = str(tmpdir / ('binary_model.cmf'))
    binz.save(model_file)

    eval_device = C.cpu()
    model = C.Function.load(model_file, device=eval_device)

    # convert to native halide implementation.
    native_binz = qc.convert_to_native_binary_convolution(model)

    functions = C.logging.graph.depth_first_search(
                native_binz, (lambda x : type(x) == C.Function and x.op_name =='BinaryConvolveOp') , depth = 0)
    assert(len(functions) == 3)

    img_data = np.reshape(dat, (1, 1, 28, 28))

    res = native_binz.eval(img_data, device=eval_device)
    assert(len(res) > 0) # evaluation should work with the new model.
