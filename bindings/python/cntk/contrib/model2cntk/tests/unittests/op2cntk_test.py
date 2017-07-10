import pytest
import cntk
import caffe
import numpy as np
from google.protobuf import text_format
from unimodel.cntkmodel import *
from adapter.bvlccaffe.caffeadapter import SetupCaffeParameters, CaffeAdapter
from unimodel.cntkinstance import ApiSetup

# global setting
precision = np.float32
# device_id = 0


# --------- util funcs -----------
def load_caffe_layer(caffe_proto):
    caffe_net = caffe.proto.caffe_pb2.NetParameter()
    text_format.Merge(caffe_proto, caffe_net)
    caffe_layers = caffe_net.layer or caffe_net.layers
    return caffe_layers[:-1], caffe_layers[-1]


def to_cntk_layer(caffe_inputs, caffe_layer, layer_type):
    data_providers = dict()
    for caffe_in in caffe_inputs:
        assert caffe_in.type == 'Input'
        for idx, top in enumerate(caffe_in.top):
            data_provider = CntkLayersDefinition()
            data_provider.op_name = top
            # here delete minibatch dims
            data_provider.tensor = caffe_in.input_param.shape[idx].dim[1:]
            data_providers[top] = data_provider

    bottoms = caffe_layer.bottom
    bottom_layers = dict()
    for bottom in bottoms:
        bottom_layers[bottom] = data_providers[bottom]

    cntk_layer = CntkLayersDefinition()
    cntk_layer.op_name = caffe_layer.name
    cntk_layer.op_type = layer_type
    for in_name in bottom_layers.keys():
        cntk_layer.inputs.append(in_name)
    cntk_layer.outputs.append(caffe_layer.name)

    return cntk_layer, bottom_layers, data_providers


def setup_layer_inputs(cntk_layer, data_providers):
    layer_inputs = list()
    for op_name, provider in data_providers.items():
        if op_name in cntk_layer.inputs:
            input_func = cntk.input(tuple(provider.tensor[:]), name=op_name)
            layer_inputs.append(input_func)
    return layer_inputs


def set_layer_weights(cntk_layer, weight_maps):
    global precision
    for w_map in weight_maps:
        w_map = np.asarray(w_map, dtype=precision)
        tensor = CntkTensorDefinition()
        tensor.data = w_map
        tensor.tensor = w_map.shape
        cntk_layer.parameter_tensor.append(tensor)


def layer_tester(layer, data, expected_out, expand_data=None):
    global precision

    np_data = np.asarray(data, dtype=precision)
    if not expand_data:
        out = layer(np_data)
    else:
        np_expand_data = np.asarray(expand_data, dtype=precision)
        out = layer(np_data, np_expand_data)

    assert (out == np.asarray(expected_out, dtype=precision)).all()
    return out


# --- for debug ---
def debug_output(cntk_layer, setup_layer, test_data, kernel_data, bias_data, out):
    print('--------------------cntk_layer info--------------------')
    print('op_name: ', cntk_layer.op_name)
    print('op_type: ', cntk_layer.op_type)
    print('inputs : ', cntk_layer.inputs)
    print('outputs: ', cntk_layer.outputs)
    print('tensor : ', cntk_layer.tensor)
    print('params : ', [i.data for i in cntk_layer.parameter_tensor])
    print('----------------------cntk_op info----------------------')
    print('cntk_op: ', setup_layer)
    print('input  : ', test_data)
    print('kernel : ', kernel_data)
    print('bias   : ', bias_data)
    print('output : ', out)
    print('--------------------------------------------------------')

# -------- unit test --------

CONV_TEST_DATA = [
    (
        ('name: "conv-test"                  '
         'layer {                            '
         '    name: "input"                  '
         '    type: "Input"                  '
         '    top : "data"                   '
         '    input_param {                  '
         '        shape {                    '
         '            dim: 1                 '
         '            dim: 1                 '
         '            dim: 2                 '
         '            dim: 2                 '
         '        }                          '
         '    }                              '
         '}                                  '
         'layer {                            '
         '    name:   "conv"                 '
         '    type:   "Convolution"          '
         '    bottom: "data"                 '
         '    convolution_param {            '
         '        num_output: 1              '
         '        bias_term: true            '
         '        pad: 0                     '
         '        kernel_size: 2             '
         '        stride: 1                  '
         '    }                              '
         '}                                  '),
        [[[[1., 2.],
           [7., 8.]]]],             # input
        [[[[71.]]]],                # output
        [[[[5., 6.],
           [3., 4.]]]],             # kernel map
        [[1., ]],                   # bias map
    )
]


@pytest.mark.parametrize("caffe_proto, test_data, expected_out, kernel_map, bias_map", CONV_TEST_DATA)
def test_convolution_setup(caffe_proto, test_data, expected_out, kernel_map, bias_map):
    caffe_inputs, caffe_layer = load_caffe_layer(caffe_proto)
    cntk_layer, bottom_layers, data_providers = to_cntk_layer(caffe_inputs, caffe_layer, CntkLayerType.convolution)

    SetupCaffeParameters.convolution(
        CaffeAdapter.get_layer_parameters(caffe_layer), bottom_layers.values(), cntk_layer)

    weight_maps = [kernel_map]
    if cntk_layer.parameters.need_bias:
        weight_maps.append(bias_map)
    set_layer_weights(cntk_layer, weight_maps)

    layer_inputs = setup_layer_inputs(cntk_layer, data_providers)
    conv_layer = ApiSetup.convolution(cntk_layer, layer_inputs)

    out = layer_tester(conv_layer, test_data, expected_out)

    debug_output(cntk_layer, conv_layer, test_data, kernel_map, bias_map, out)


POOL_TEST_DATA = [
    (
        ('name: "pool-test"                  '
         'layer {                            '
         '    name: "input"                  '
         '    type: "Input"                  '
         '    top : "data"                   '
         '    input_param {                  '
         '        shape {                    '
         '            dim: 1                 '
         '            dim: 1                 '
         '            dim: 6                 '
         '            dim: 6                 '
         '        }                          '
         '    }                              '
         '}                                  '
         'layer {                            '
         '    name:   "pool"                 '
         '    type:   "Pooling"              '
         '    bottom: "data"                 '
         '    top:    "pool"                 '
         '    pooling_param {                '
         '        pool: MAX                  '
         '        kernel_size: 3             '
         '        stride: 3                  '
         '    }                              '
         '}                                  '),
        [[[[0.,   1.,  2.,  3.,  4.,  5.],
           [6.,   7.,  8.,  9., 10., 11.],
           [12., 13., 14., 15., 16., 17.],
           [18., 19., 20., 21., 22., 23.],
           [24., 25., 26., 27., 28., 29.],
           [30., 31., 32., 33., 34., 35.]]]],
        [[[[14., 17.],
           [32., 35.]]]]
    )
]


@pytest.mark.parametrize("caffe_proto, test_data, expected_out", POOL_TEST_DATA)
def test_pooling_setup(caffe_proto, test_data, expected_out):
    caffe_inputs, caffe_layer = load_caffe_layer(caffe_proto)
    cntk_layer, bottom_layers, data_providers = to_cntk_layer(caffe_inputs, caffe_layer, CntkLayerType.pooling)

    SetupCaffeParameters.pooling(
        CaffeAdapter.get_layer_parameters(caffe_layer), bottom_layers.values(), cntk_layer)

    layer_inputs = setup_layer_inputs(cntk_layer, data_providers)
    pool_layer = ApiSetup.pooling(cntk_layer, layer_inputs)

    out = layer_tester(pool_layer, test_data, expected_out)

    debug_output(cntk_layer, pool_layer, test_data, None, None, out)


BATCHNORM_TEST_DATA = [
    (
        ('name: "batchnorm-test"             '
         'layer {                            '
         '    name: "input"                  '
         '    type: "Input"                  '
         '    top : "data"                   '
         '    input_param {                  '
         '        shape {                    '
         '            dim: 1                 '
         '            dim: 2                 '
         '            dim: 2                 '
         '            dim: 2                 '
         '        }                          '
         '    }                              '
         '}                                  '
         'layer {                            '
         '    name:   "batchnorm"            '
         '    type:   "BatchNorm"            '
         '    bottom: "data"                 '
         '    top:    "batchnorm"            '
         '    batch_norm_param {             '
         '        use_global_stats: true     '
         '    }                              '
         '}                                  '),
        [[[[1., 2.], [3., 4.]],
          [[2., 1.], [4., 3.]]]],
        [[[[316.2277832, 632.45556641],
           [948.68328857, 1264.91113281]],
          [[632.45556641, 316.2277832],
           [1264.91113281, 948.68328857]]]],
        [0, 0],     #0 run_mean
        [0, 0],     #1 run_var
        [0],        #2 global_scale
        [1, 1],     #3 scale
        [0, 0],     #4 bias
    )
]


@pytest.mark.parametrize("caffe_proto, test_data, expected_out, mean, var, g_scale, scale, bias", BATCHNORM_TEST_DATA)
def test_batchnorm_setup(caffe_proto, test_data, expected_out, mean, var, g_scale, scale, bias):
    caffe_inputs, caffe_layer = load_caffe_layer(caffe_proto)
    cntk_layer, bottom_layers, data_providers = \
        to_cntk_layer(caffe_inputs, caffe_layer, CntkLayerType.batch_normalization)

    SetupCaffeParameters.batch_normalization(
        CaffeAdapter.get_layer_parameters(caffe_layer), bottom_layers.values(), cntk_layer)

    weight_maps = [mean, var, g_scale, scale, bias]
    set_layer_weights(cntk_layer, weight_maps)

    layer_inputs = setup_layer_inputs(cntk_layer, data_providers)
    batchnorm_layer = ApiSetup.batch_normalization(cntk_layer, layer_inputs)

    out = layer_tester(batchnorm_layer, test_data, expected_out)

    debug_output(cntk_layer, batchnorm_layer, test_data, mean, var, out)


RELU_TEST_DATA = [
    (
        (
         'name: "relu-test"                  '
         'layer {                            '
         '    name: "input"                  '
         '    type: "Input"                  '
         '    top : "data"                   '
         '    input_param {                  '
         '        shape {                    '
         '            dim: 1                 '
         '            dim: 1                 '
         '            dim: 3                 '
         '            dim: 3                 '
         '        }                          '
         '    }                              '
         '}                                  '
         'layer {                            '
         '    name:   "relu"                 '
         '    type:   "ReLU"                 '
         '    bottom: "data"                 '
         '    top:    "relu"                 '
         '}                                  '),
        [[[[ 0., -1.,  2.],
           [-3.,  4., -5.],
           [-6.,  7.,  8.]]]],
        [[[[0., 0., 2.],
           [0., 4., 0.],
           [0., 7., 8.]]]],
    ),
]


@pytest.mark.parametrize("caffe_proto, test_data, expected_out", RELU_TEST_DATA)
def test_relu_setup(caffe_proto, test_data, expected_out):
    caffe_inputs, caffe_layer = load_caffe_layer(caffe_proto)
    cntk_layer, bottom_layers, data_providers = to_cntk_layer(caffe_inputs, caffe_layer, CntkLayerType.relu)

    SetupCaffeParameters.relu(
        CaffeAdapter.get_layer_parameters(caffe_layer), bottom_layers.values(), cntk_layer)

    layer_inputs = setup_layer_inputs(cntk_layer, data_providers)
    relu_layer = ApiSetup.relu(cntk_layer, layer_inputs)

    out = layer_tester(relu_layer, test_data, expected_out)

    debug_output(cntk_layer, relu_layer, test_data, None, None, out)


DENSE_TEST_DATA = [
    (
        ('name: "dense-test"                 '
         'layer {                            '
         '    name: "input"                  '
         '    type: "Input"                  '
         '    top : "data"                   '
         '    input_param {                  '
         '        shape {                    '
         '            dim: 1                 '
         '            dim: 1                 '
         '            dim: 1                 '
         '            dim: 3                 '
         '        }                          '
         '    }                              '
         '}                                  '
         'layer {                            '
         '    name:   "fc"                   '
         '    type:   "InnerProduct"         '
         '    bottom: "data"                 '
         '    top:    "fc"                   '
         '    inner_product_param {          '
         '        num_output: 5              '
         '    }                              '
         '}                                  '),
        [[[[1., 2., 3.]]]],
        [[[[7., 14, 21, 28, 35]]]],
        [[[[1., 1., 1.],
           [2., 2., 2.],
           [3., 3., 3.],
           [4., 4., 4.],
           [5., 5., 5.]]]],
        [[[[1., 2., 3., 4., 5.]]]],
    )
]


@pytest.mark.parametrize("caffe_proto, test_data, expected_out, scale, bias", DENSE_TEST_DATA)
def test_dense_setup(caffe_proto, test_data, expected_out, scale, bias):
    caffe_inputs, caffe_layer = load_caffe_layer(caffe_proto)
    cntk_layer, bottom_layers, data_providers = to_cntk_layer(caffe_inputs, caffe_layer, CntkLayerType.dense)

    SetupCaffeParameters.dense(
        CaffeAdapter.get_layer_parameters(caffe_layer), bottom_layers.values(), cntk_layer)

    weights_maps = [scale, bias]
    set_layer_weights(cntk_layer, weights_maps)

    layer_inputs = setup_layer_inputs(cntk_layer, data_providers)
    dense_layer = ApiSetup.dense(cntk_layer, layer_inputs)

    out = layer_tester(dense_layer, test_data, expected_out)

    debug_output(cntk_layer, dense_layer, test_data, scale, bias, out)


PLUS_TEST_DATA = [
    (
        ('name: "plus-test"                 '
         'layer {                            '
         '    name: "input"                  '
         '    type: "Input"                  '
         '    top : "data1"                  '
         '    input_param {                  '
         '        shape {                    '
         '            dim: 1                 '
         '            dim: 1                 '
         '            dim: 3                 '
         '            dim: 3                 '
         '        }                          '
         '    }                              '
         '}                                  '
         'layer {                            '
         '    name: "input"                  '
         '    type: "Input"                  '
         '    top : "data2"                  '
         '    input_param {                  '
         '        shape {                    '
         '            dim: 1                 '
         '            dim: 1                 '
         '            dim: 3                 '
         '            dim: 3                 '
         '        }                          '
         '    }                              '
         '}                                  '
         'layer {                            '
         '    name:   "eltwise"              '
         '    type:   "Eltwise"              '
         '    bottom: "data1"                '
         '    bottom: "data2"                '
         '    top:    "eltwise"              '
         '}                                  '),
        ([[[[1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]]]],
         [[[[1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]]]]),
        [[[[ 2.,  4.,  6.],
           [ 8., 10.,  12.],
           [14., 16.,  18.]]]],
    )
]


@pytest.mark.parametrize("caffe_proto, test_data, expected_out", PLUS_TEST_DATA)
def test_plus_setup(caffe_proto, test_data, expected_out):
    caffe_inputs, caffe_layer = load_caffe_layer(caffe_proto)
    cntk_layer, bottom_layers, data_providers = to_cntk_layer(caffe_inputs, caffe_layer, CntkLayerType.plus)

    SetupCaffeParameters.plus(
        CaffeAdapter.get_layer_parameters(caffe_layer), bottom_layers.values(), cntk_layer)

    layer_inputs = setup_layer_inputs(cntk_layer, data_providers)
    plus_layer = ApiSetup.plus(cntk_layer, layer_inputs)

    l_data, r_data = test_data[0], test_data[1]
    out = layer_tester(plus_layer, l_data, expected_out, r_data)

    debug_output(cntk_layer, plus_layer, test_data, None, None, out)


CROSS_ENTROPY_WITH_SOFTMAX_TEST_DATA = [
    (
        ('name: "ce_s-test"                  '
         'layer {                            '
         '    name: "input"                  '
         '    type: "Input"                  '
         '    top : "data1"                  '
         '    input_param {                  '
         '        shape {                    '
         '            dim: 1                 '
         '            dim: 4                 '
         '        }                          '
         '    }                              '
         '}                                  '
         'layer {                            '
         '    name: "input"                  '
         '    type: "Input"                  '
         '    top : "data2"                  '
         '    input_param {                  '
         '        shape {                    '
         '            dim: 1                 '
         '            dim: 4                 '
         '        }                          '
         '    }                              '
         '}                                  '
         'layer {                            '
         '    name:   "softmax_with_loss"    '
         '    type:   "SoftmaxWithLoss"      '
         '    bottom: "data1"                '
         '    bottom: "data2"                '
         '    top:    "swl"                  '
         '}                                  '),
        ([[1., 1., 1., 50.]],
         [[0., 0., 0., 1.]]),
        [[0.]],

    )
]


@pytest.mark.parametrize("caffe_proto, test_data, expected_out", CROSS_ENTROPY_WITH_SOFTMAX_TEST_DATA)
def test_cross_entropy_with_softmax_setup(caffe_proto, test_data, expected_out):
    caffe_inputs, caffe_layer = load_caffe_layer(caffe_proto)
    cntk_layer, bottom_layers, data_providers = to_cntk_layer(caffe_inputs, caffe_layer, CntkLayerType.plus)

    SetupCaffeParameters.cross_entropy_with_softmax(
        CaffeAdapter.get_layer_parameters(caffe_layer), bottom_layers.values(), cntk_layer)

    layer_inputs = setup_layer_inputs(cntk_layer, data_providers)
    ces_layer = ApiSetup.cross_entropy_with_softmax(cntk_layer, layer_inputs)

    x_data, y_data = test_data[0], test_data[1]
    out = layer_tester(ces_layer, x_data, expected_out, y_data)

    debug_output(cntk_layer, ces_layer, test_data, None, None, out)


DROPOUT_TEST_DATA = [
    (
        ('name: "dropout-test"               '
         'layer {                            '
         '    name: "input"                  '
         '    type: "Input"                  '
         '    top : "data"                   '
         '    input_param {                  '
         '        shape {                    '
         '            dim: 1                 '
         '            dim: 1                 '
         '            dim: 3                 '
         '            dim: 2                 '
         '        }                          '
         '    }                              '
         '}                                  '
         'layer {                            '
         '    name:   "dropout"              '
         '    type:   "Dropout"              '
         '    bottom: "data"                 '
         '    top:    "dropout"              '
         '}                                  '),
        [[[[10, 20],
           [30, 40],
           [50, 60]]]],
        [[[[10, 20],
           [30, 40],
           [50, 60]]]],
    )
]


@pytest.mark.parametrize("caffe_proto, test_data, expected_out", DROPOUT_TEST_DATA)
def test_dropout_setup(caffe_proto, test_data, expected_out):
    caffe_inputs, caffe_layer = load_caffe_layer(caffe_proto)
    cntk_layer, bottom_layers, data_providers = to_cntk_layer(caffe_inputs, caffe_layer, CntkLayerType.plus)

    SetupCaffeParameters.dropout(
        CaffeAdapter.get_layer_parameters(caffe_layer), bottom_layers.values(), cntk_layer)

    layer_inputs = setup_layer_inputs(cntk_layer, data_providers)
    dropout_layer = ApiSetup.dropout(cntk_layer, layer_inputs)

    out = layer_tester(dropout_layer, test_data, expected_out)

    debug_output(cntk_layer, dropout_layer, test_data, None, None, out)


LRN_TEST_DATA = [
    (
        ('name: "lrn-test"                   '
         'layer {                            '
         '    name: "input"                  '
         '    type: "Input"                  '
         '    top : "data"                   '
         '    input_param {                  '
         '        shape {                    '
         '            dim: 1                 '
         '            dim: 1                 '
         '            dim: 3                 '
         '            dim: 3                 '
         '        }                          '
         '    }                              '
         '}                                  '
         'layer {                            '
         '    name:   "lrn"                  '
         '    type:   "LRN"                  '
         '    bottom: "data"                 '
         '    top:    "lrn"                  '
         '    lrn_param {                    '
         '        local_size: 1              '
         '        alpha: 1                   '
         '        beta: 0.75                 '
         '        k: 1                       '
         '    }                              '
         '}                                  '),
        [[[[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]]]],
        [[[[0.59460354, 0.59460354, 0.59460354],
           [0.59460354, 0.59460354, 0.59460354],
           [0.59460354, 0.59460354, 0.59460354]]]],
    )
]


@pytest.mark.parametrize("caffe_proto, test_data, expected_out", LRN_TEST_DATA)
def test_lrn_setup(caffe_proto, test_data, expected_out):
    caffe_inputs, caffe_layer = load_caffe_layer(caffe_proto)
    cntk_layer, bottom_layers, data_providers = to_cntk_layer(caffe_inputs, caffe_layer, CntkLayerType.plus)

    SetupCaffeParameters.lrn(
        CaffeAdapter.get_layer_parameters(caffe_layer), bottom_layers.values(), cntk_layer)

    layer_inputs = setup_layer_inputs(cntk_layer, data_providers)
    lrn_layer = ApiSetup.lrn(cntk_layer, layer_inputs)

    out = layer_tester(lrn_layer, test_data, expected_out)

    debug_output(cntk_layer, lrn_layer, test_data, None, None, out)


SPLICE_TEST_DATA = [
    (
        ('name: "splice-test"                '
         'layer {                            '
         '    name: "input"                  '
         '    type: "Input"                  '
         '    top : "data1"                  '
         '    input_param {                  '
         '        shape {                    '
         '            dim: 1                 '
         '            dim: 1                 '
         '            dim: 2                 '
         '            dim: 2                 '
         '        }                          '
         '    }                              '
         '}                                  '
         'layer {                            '
         '    name: "input"                  '
         '    type: "Input"                  '
         '    top : "data2"                  '
         '    input_param {                  '
         '        shape {                    '
         '            dim: 1                 '
         '            dim: 1                 '
         '            dim: 3                 '
         '            dim: 2                 '
         '        }                          '
         '    }                              '
         '}                                  '
         'layer {                            '
         '    name: "concat"                 '
         '    type: "Concat"                 '
         '    bottom: "data1"                '
         '    bottom: "data2"                '
         '    top: "concat"                  '
         '    concat_param {                 '
         '        axis: 2                    '
         '    }                              '
         '}                                  '),
        ([[[[1., 2.],
            [4., 5.]]]],

         [[[[10., 20.],
            [30., 40.],
            [50., 60.]]]]),

        [[[[1., 2.],
           [4., 5.],
           [10., 20.],
           [30., 40.],
           [50., 60.]]]]
    )
]


@pytest.mark.parametrize("caffe_proto, test_data, expected_out", SPLICE_TEST_DATA)
def test_splice_setup(caffe_proto, test_data, expected_out):
    caffe_inputs, caffe_layer = load_caffe_layer(caffe_proto)
    cntk_layer, bottom_layers, data_providers = to_cntk_layer(caffe_inputs, caffe_layer, CntkLayerType.plus)

    SetupCaffeParameters.splice(
        CaffeAdapter.get_layer_parameters(caffe_layer), bottom_layers.values(), cntk_layer)

    layer_inputs = setup_layer_inputs(cntk_layer, data_providers)
    splice_layer = ApiSetup.splice(cntk_layer, layer_inputs)

    data1, data2 = test_data[0], test_data[1]
    out = layer_tester(splice_layer, data1, expected_out, data2)

    debug_output(cntk_layer, splice_layer, test_data, None, None, out)


SOFTMAX_TEST_DATA = [
    (
        ('name: "softmax-test"               '
         'layer {                            '
         '    name: "input"                  '
         '    type: "Input"                  '
         '    top : "data"                   '
         '    input_param {                  '
         '        shape {                    '
         '            dim: 1                 '
         '            dim: 1                 '
         '            dim: 1                 '
         '            dim: 2                 '
         '        }                          '
         '    }                              '
         '}                                  '
         'layer {                            '
         '    name: "softmax"                '
         '    type: "Softmax"                '
         '    bottom: "data"                 '
         '    top: "softmax"                 '
         '}                                  '),
        [[[[1, 1]]]],
        [[[[0.5,  0.5]]]],
    )
]


@pytest.mark.parametrize("caffe_proto, test_data, expected_out", SOFTMAX_TEST_DATA)
def test_softmax_setup(caffe_proto, test_data, expected_out):
    caffe_inputs, caffe_layer = load_caffe_layer(caffe_proto)
    cntk_layer, bottom_layers, data_providers = to_cntk_layer(caffe_inputs, caffe_layer, CntkLayerType.plus)

    SetupCaffeParameters.softmax(
        CaffeAdapter.get_layer_parameters(caffe_layer), bottom_layers.values(), cntk_layer)

    layer_inputs = setup_layer_inputs(cntk_layer, data_providers)
    softmax_layer = ApiSetup.softmax(cntk_layer, layer_inputs)

    out = layer_tester(softmax_layer, test_data, expected_out)

    debug_output(cntk_layer, softmax_layer, test_data, None, None, out)
