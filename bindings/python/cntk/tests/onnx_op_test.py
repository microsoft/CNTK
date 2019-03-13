# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import numpy as np
import cntk as C
import pytest
onnx = pytest.importorskip("onnx")
from copy import deepcopy
from cntk.ops.tests.ops_test_utils import cntk_device
from itertools import product
from .onnx_test_helper import transpose_dynamic_axis, create_and_populate_onnx_test_case_with_model_conversion, save_test_data, compare_model_for_output_data_transpose
from .onnx_test_helper import CNTK_FREEDIM_AXIS_DENOTATION, DIM_SIZE_FOR_NON_BATCH_OPS, is_list_of_sparse, sparse_to_dense

# This is a list of all ops in CNTK that are exported as ONNX ops
# that have a batch axis defined by spec (e.g. convolution, pooling)
# When adding a test for a new op, please check to see if 
# that op needs to be added to this list (i.e. does that op 
# get exported to an ONNX op with defined batch axis).
set_of_batch_ops = {'Pooling', 'Convolution', 'GlobalAveragePooling', 'GlobalMaxPooling', 'DepthToSpace', 'SpaceToDepth', 'LocalResponseNormalization', 'MeanVarianceNormalization', 'LayerNormalization', 'BatchNormalization', 'ImageScaler', 'Crop'}

# List of CNTK ops for which output shape doesn't change regardless
# of whether the input has batch axis or not.
# Basically, for these ops we don't prepend 1 to the output shape
# when the input has batch axis.
set_of_batch_irrelevant_ops = {}

##########################################
## helper verification functions
##########################################

def init_empty_node_names(model):
    # Most of the unit tests here don't specify names for nodes.
    # Try to replace empty node names, and check if names are preserved after export/import
    # in later tests.
    class UpdateNodeName(object):
        i = 0
        @classmethod
        def step(cls, node):
            if node.name == "":
                try:
                    node.name = "test_node_name_" + str(cls.i)
                    cls.i += 1
                except:
                    return True
            return True
    C.logging.graph.depth_first_search(model, UpdateNodeName.step)

def verify_node_names(model1, model2):
    # Verify if all the node names in original model appears at least as prefix of some node names
    # in reloaded model. Because alternations of nodes are occasionally necessary in exporting and importing,
    # in these cases node names might be appended with different postfixes.
    model1_names = [node.name for node in C.logging.graph.depth_first_search(model1, lambda x : True)]
    model2_names = [node.name for node in C.logging.graph.depth_first_search(model2, lambda x : True)]

    names_preserved = [name == '' or any([new_name.startswith(name) for new_name in model2_names])
                       for name in model1_names]
    assert all(names_preserved) == True

def verify_no_input(model, tmpdir, name, use_external_files_to_store_parameters = False):
    init_empty_node_names(model)

    opname = model.owner.op_name

    loaded_model = None
    loaded_model, _, _, _ = create_and_populate_onnx_test_case_with_model_conversion(
        model, tmpdir, name, loaded_model, 
        use_external_files_to_store_parameters = use_external_files_to_store_parameters)

    model_shape = model.shape
    dim_denotation = None
    if model.output.dynamic_axes == (C.Axis('defaultBatchAxis'),) and opname not in set_of_batch_ops:
        dim_denotation = DIM_SIZE_FOR_NON_BATCH_OPS
    elif opname in set_of_batch_ops:
        dim_denotation = CNTK_FREEDIM_AXIS_DENOTATION
    if not dim_denotation is None and opname not in set_of_batch_irrelevant_ops:
        model_shape = (dim_denotation, ) + model_shape

    assert model_shape == loaded_model.shape

    o0 = model.eval()
    o1 = loaded_model.eval()

    if (type(o0) is list):
        o0 = o0[0]
    if (type(o1) is list):
        o1 = o1[0]

    assert np.allclose(o0, o1)
    verify_node_names(model, loaded_model)
    return loaded_model

def verify_one_input(model, data, tmpdir, name, device=None, loaded_model=None, rtol = 1e-05, atol = 1e-08, 
                     bypass_load_into_cntk = False, use_external_files_to_store_parameters = False):
    # TODO: eventually we want this test method to be more general to suport 
    # models with multiple inputs instead of just one input.
    assert len(model.arguments) == 1
    assert not model.arguments[0].has_sequence_axis()

    init_empty_node_names(model)

    # data here is reference to the outside data object. create deepcopy to avoid changing the outside data since it might get reused.
    data = deepcopy(data)

    # outputs share the same owner
    opname = model.outputs[0].owner.op_name

    if bypass_load_into_cntk:
        loaded_model, onnx_model, test_model_path, test_data_path = create_and_populate_onnx_test_case_with_model_conversion(
            model, tmpdir, name, model, bypass_load_into_cntk=True,
            use_external_files_to_store_parameters = use_external_files_to_store_parameters)
    else:
        loaded_model, onnx_model, test_model_path, test_data_path = create_and_populate_onnx_test_case_with_model_conversion(
            model, tmpdir, name, loaded_model, 
            use_external_files_to_store_parameters = use_external_files_to_store_parameters)

    # TODO: it is better to compare data.shape with model.arguments[0] and
    # to pad batch dimension as needed.
    # Some tests have already expanded batch axis to data (i.e. reduction test) 
    if model.arguments[0].has_batch_axis() and type(data)!=list:
        data.shape = (1, ) + data.shape

    if not bypass_load_into_cntk:
        assert len(model.outputs) == len(loaded_model.outputs)

    dim_denotation = CNTK_FREEDIM_AXIS_DENOTATION if opname in set_of_batch_ops else DIM_SIZE_FOR_NON_BATCH_OPS
    for i in range(0, len(model.outputs)):
        assert not model.outputs[i].has_sequence_axis()
        output_shape = model.outputs[i].shape
        if opname not in set_of_batch_irrelevant_ops:
            if model.outputs[i].has_batch_axis():
                output_shape = (dim_denotation, ) + output_shape
        if not bypass_load_into_cntk:
            assert output_shape == loaded_model.outputs[i].shape

    if device:
        o0 = model.eval({model.arguments[0]:data}, device=device)
        o1 = loaded_model.eval({loaded_model.arguments[0]:data}, device=device)
    else:
        o0 = model.eval({model.arguments[0]:data})
        o1 = loaded_model.eval({loaded_model.arguments[0]:data})

    if len(model.outputs) == 1:
        assert np.allclose(o0, o1, rtol, atol)
    else:
        matched_indices = []
        for i in range(0, len(model.outputs)):
            # outputs of loaded model are not necessarily in the same order as the original model.
            # output uid is likely changed too.
            # the only way to verify the data is to find match for every output. 
            o0i = o0[model.outputs[i]]
            for j in range(0, len(loaded_model.outputs)):
                if j not in matched_indices:
                    o1i = o1[loaded_model.outputs[j]]
                    if np.shape(o0i) == np.shape(o1i) and np.allclose(o0i, o1i):
                        matched_indices.append(j)
                        break
            assert len(matched_indices) == i+1

    save_test_data(model, onnx_model, test_data_path, data, o0, name, tmpdir)

    verify_node_names(model, loaded_model)
    return loaded_model

def run_model(model, data, device=None):
    feed = {}
    if len(model.arguments) == 1:
        feed[model.arguments[0]] = data
    elif len(model.arguments) > 1:
        assert len(model.arguments) == len(data)
        for i in range(len(model.arguments)):
            feed[model.arguments[i]] = data[i]
            
    o = model.eval(feed, device=device)
    return o

def verify_sequence_model(model, data, tmpdir, name, device=None, loaded_model=None, resave = True, bypass_load_into_cntk = False,
                          use_external_files_to_store_parameters = False):
    # data here is reference to the outside data object. create deepcopy to avoid changing the outside data since it might get reused.
    data = deepcopy(data)

    # onnx does not specify sparse tensor. to run imported model, a sparse matrix needs to be converted to a dense matrix 
    if bypass_load_into_cntk:
        dataOnnx = data
        loaded_model, onnx_model, test_model_path, test_data_path = create_and_populate_onnx_test_case_with_model_conversion(
            model, tmpdir, name, model, resave, True, 
            use_external_files_to_store_parameters = use_external_files_to_store_parameters)
        o0 = run_model(model, data, device=device)
        o0 = np.array(o0)
        save_test_data(model, onnx_model, test_data_path, data, o0, name, tmpdir)
    else:
        dataOnnx = None
        if is_list_of_sparse(data):
            dataOnnx = transpose_dynamic_axis(sparse_to_dense(data))
        else:
            if (type(data) == list):
                dataOnnx = []
                for i in range(0, len(data)):
                    if (model.arguments[i].has_sequence_axis()):
                        dataOnnx.append(transpose_dynamic_axis(data[i]))
                    else:
                        dataOnnx.append(data[i])
            else:
                dataOnnx = transpose_dynamic_axis(data)

        loaded_model, onnx_model, test_model_path, test_data_path = create_and_populate_onnx_test_case_with_model_conversion(
            model, tmpdir, name, loaded_model, resave,
            use_external_files_to_store_parameters = use_external_files_to_store_parameters)

        o0 = run_model(model, data, device=device)
        o1 = run_model(loaded_model, dataOnnx, device=device)

        ## if there is a sequence axis in the output, it must be swapped with batch axis 
        ## to match the original CNTK model's output 
        if len(model.outputs) == 1:
            o0 = np.array(o0)
            o1 = np.array(o1)
            if compare_model_for_output_data_transpose(model.outputs[0], loaded_model.outputs[0]):
                o1 = transpose_dynamic_axis(np.array(o1))
            assert np.allclose(o0, o1)
        else:
            matched_indices = []
            for i in range(0, len(model.outputs)):
                # outputs of loaded model are not necessarily in the same order as the original model.
                # output uid is likely changed too.
                # the only way to verify the data is to find match for every output. 
                o0i = o0[model.outputs[i]]
                for j in range(0, len(loaded_model.outputs)):
                    if j not in matched_indices:
                        o1i = o1[loaded_model.outputs[j]]
                        if compare_model_for_output_data_transpose(model.outputs[i], loaded_model.outputs[j]):
                            o1i = transpose_dynamic_axis(o1i)
                        if np.shape(o0i) == np.shape(o1i) and np.allclose(o0i, o1i):
                            matched_indices.append(j)
                            break
                assert len(matched_indices) == i+1
        save_test_data(model, onnx_model, test_data_path, data, o0, name, tmpdir)

def verify_two_input(model, data1, data2, tmpdir, name, use_external_files_to_store_parameters=False):
    init_empty_node_names(model)

    # data here is reference to the outside data object. create deepcopy to avoid changing the outside data since it might get reused.
    data1 = deepcopy(data1)
    data2 = deepcopy(data2)

    filename = os.path.join(str(tmpdir), name + R'.onnx')
    model.save(filename, format=C.ModelFormat.ONNX, 
               use_external_files_to_store_parameters = use_external_files_to_store_parameters)
    opname = model.owner.op_name

    loaded_model = C.Function.load(filename, format=C.ModelFormat.ONNX)

    filename_resave = os.path.join(str(tmpdir), name + R'_resave.onnx')
    loaded_model.save(filename_resave, format=C.ModelFormat.ONNX)

    model_shape = model.shape
    if model.output.dynamic_axes == (C.Axis('defaultBatchAxis'),):
        dim_denotation = CNTK_FREEDIM_AXIS_DENOTATION if opname in set_of_batch_ops else DIM_SIZE_FOR_NON_BATCH_OPS
        if opname not in set_of_batch_irrelevant_ops:
            model_shape = (dim_denotation, ) + model_shape
        data1.shape = (1, ) + data1.shape
        data2.shape = (1, ) + data2.shape
    assert model_shape == loaded_model.shape

    o0 = model.eval({model.arguments[0]:data1, model.arguments[1]:data2})
    o1 = loaded_model.eval({loaded_model.arguments[0]:data1, loaded_model.arguments[1]:data2})

    if (type(o0) is list):
        o0 = o0[0]
    if (type(o1) is list):
        o1 = o1[0]

    assert np.allclose(o0, o1)
    verify_node_names(model, loaded_model)

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
@pytest.mark.parametrize("dtype", DType_Config)
def test_And(tmpdir, dtype):
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

        # test for case of not padding but with ceilOutDim=True
        img = np.reshape(np.arange(49, dtype=dtype), [1, 7, 7])
        x = C.input_variable(img.shape)
        model = C.pooling(x, C.AVG_POOLING, (7, 7), auto_padding = [False, False, False], ceil_out_dim=True)

        verify_one_input(model, img, tmpdir, 'AveragePool_2', device)

#BatchNormalization
def verify_BN(x, init_scale, init_bias, mean, var, epsilon, spatial, tmpdir, dtype):
    with C.default_options(dtype = dtype):
        scale        = C.Parameter(init=init_scale, dtype=np.float32)
        bias         = C.Parameter(init=init_bias, dtype=np.float32)
        run_mean     = C.ops.constant(mean, shape=mean.shape, dtype=np.float32)
        run_variance = C.ops.constant(var,  shape=var.shape, dtype=np.float32)
        run_count    = C.ops.constant(0,               dtype=np.float32)

        a = C.input_variable(shape=x.shape[1:], dtype=dtype, needs_gradient=False, name='a')

        op_node = C.batch_normalization(a, scale, bias, run_mean, run_variance, running_count=run_count, spatial=spatial,
            epsilon=epsilon)

        loaded_model = None
        test_base_name = 'Spatial' if spatial else ''
        test_base_name = test_base_name + ('BatchNormalization_float16' if dtype==np.float16 else 'BatchNormalization_float32')

        for i in range(len(x)):
            if dtype==np.float16:
                loaded_model = verify_one_input(op_node, x[i], tmpdir, test_base_name + str(i), loaded_model=loaded_model, rtol = 1e-03, atol = 1e-03)
            else:
                loaded_model = verify_one_input(op_node, x[i], tmpdir, test_base_name + str(i), loaded_model=loaded_model)

non_spatial_float16_skip_message = str('Test is skipped with float16 data because CNTK ONNX importer in float16 case assumes mean/var inputs being constant.'
    'this is not always true because in CNTK non-spatial case mean/var may need to be reshaped before pass to the BN function.'
    'In general import of BatchNormalization(float16) need to be fixed to take any input as mean/var, etc.')
# Case 1 - Non-Spatial BN with More > 1 batches    
@pytest.mark.parametrize("dtype", DType_Config)
def test_BatchNormalization(tmpdir, dtype):
    if dtype == np.float16:
        pytest.skip(non_spatial_float16_skip_message)
    sample = [  # 5 samples having 4 classes
            [1, 1, 2, 3],
            [0, 0, 0, 0],
            [3, 3, 4, 4],
            [1000, 1000, 1000, 1000],
            [10000, 10000, 10000, 10000]]

    np.random.seed(1)
    x = np.array(sample).reshape(-1,1).astype(dtype)
    scale = np.array([3]).astype(np.float32)
    bias = np.array([4]).astype(np.float32)
    mean = np.array([1]).astype(np.float32)
    var = np.array([2]).astype(np.float32)
    epsilon = 0.00001

    verify_BN(x, scale, bias, mean, var, epsilon, False, tmpdir, dtype)
    
# Case 2 - Spatial BN with More > 1 batches    
@pytest.mark.parametrize("dtype", DType_Config)
def test_SpatialBatchNormalization(tmpdir, dtype):
    np.random.seed(0)
    x = np.random.randn(2, 3, 4, 5).astype(dtype)
    scale = np.random.randn(3).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)
    mean = np.random.randn(3).astype(np.float32)
    var = np.random.rand(3).astype(np.float32)
    epsilon = 1e-2

    verify_BN(x, scale, bias, mean, var, epsilon, True, tmpdir, dtype)

#Cast
Cast_Type_Config = (np.float64, np.float32, np.float16)
@pytest.mark.parametrize("from_type", Cast_Type_Config)
@pytest.mark.parametrize("to_type", Cast_Type_Config)
def test_Cast(tmpdir, from_type, to_type):
    test_name = "cast_" + from_type.__name__ + "_to_" + to_type.__name__
    shape = (3, 10, 15)
    input_var = C.input_variable(shape, dtype = from_type, name='features') 
    model = C.cast(input_var, dtype=to_type)
    data = np.random.rand(*shape).astype(from_type)
    verify_one_input(model, data, tmpdir, test_name)

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
def test_Concat_With_Broadcast(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        # TODO: add test cast with exchanged shape1 and shape2
        shape1 = [2,3,1,1,3]
        shape2 =   [1,3,4,1]
        shape3 =     [2,4,1]
        axis = 2
        data1 = np.random.uniform(-10, 10, shape1).astype(dtype)
        data2 = np.random.uniform(-10, 10, shape2).astype(dtype)
        data3 = np.random.uniform(-10, 10, shape3).astype(dtype)
        x = C.input_variable(shape1)
        y = C.constant(value=data2)
        z = C.constant(value=data3)
        model = C.splice(x, y, z, axis=axis)
        verify_one_input(model, data1, tmpdir, 'Concat_Braodcast')

@pytest.mark.parametrize("dtype", DType_Config)
def test_Conv(tmpdir, dtype, device_id):
    if device_id == -1 and dtype == np.float16:
        pytest.skip('Test is skipped on CPU with float16 data')
    device = cntk_device(device_id)
    with C.default_options(dtype=dtype):
        input_shape = (3, 20, 32) 
        img = np.reshape(np.arange(np.prod(input_shape), dtype = dtype), input_shape) 

        x = C.input_variable(input_shape)

        kernel_shape = (64, 3, 3, 3) # For convolution the shape is (O x I x W x H)
        kernel = C.constant(value = np.ones(shape=(kernel_shape), dtype = dtype))

        conv_model = C.convolution(kernel, x, auto_padding = [False, True, True])

        verify_one_input(conv_model, img, tmpdir, 'Conv_0', device)

@pytest.mark.parametrize("dtype", DType_Config)
def test_Conv_SpecialCase(tmpdir, dtype, device_id):
    if device_id == -1 and dtype == np.float16:
        pytest.skip('Test is skipped on CPU with float16 data')
    device = cntk_device(device_id)
    with C.default_options(dtype=dtype):
        input_shape = (3, 20, 32) 
        img = np.reshape(np.arange(np.prod(input_shape), dtype = dtype), input_shape) 

        x = C.input_variable(input_shape)

        kernel_shape = (3, 3, 3) # For convolution the shape is (O x I x W x H). Here O is omitted which is often the case in CNTK models. 
        kernel = C.constant(value = np.ones(shape=(kernel_shape), dtype = dtype))

        conv_model = C.convolution(kernel, x, auto_padding = [False, True, True])

        verify_one_input(conv_model, img, tmpdir, 'Conv_1', device)

        kernel = C.Parameter((kernel_shape), init=C.glorot_uniform(), dtype=dtype, device=device)
        conv_model = C.convolution(kernel, x, auto_padding = [False, True, True])

        verify_one_input(conv_model, img, tmpdir, 'Conv_2', device)

@pytest.mark.parametrize("dtype", DType_Config)
def test_Conv_SpecialCase_Autopad(tmpdir, dtype, device_id):
    if device_id == -1 and dtype == np.float16:
        pytest.skip('Test is skipped on CPU with float16 data')
    elif dtype == np.float16:
        pytest.skip('Test is skipped on GPU with float16 data: asymmetric padding not supported by cuDnn')
    device = cntk_device(device_id)
    with C.default_options(dtype=dtype):
        # special case where for one axis CNTK pads upper, for other lower.
        input_shape = (3, 7, 8)
        img = np.reshape(np.arange(np.prod(input_shape), dtype=dtype), input_shape)
        x = C.input_variable(input_shape)

        kernel_shape = (3, 2, 3)
        kernel = C.constant(value = np.ones(shape=(kernel_shape), dtype=dtype))
        strides = (1, 2)

        conv_model = C.convolution(kernel, x, auto_padding = [False, True, True], strides=strides)
        verify_one_input(conv_model, img, tmpdir, 'Conv_3', device)


@pytest.mark.parametrize("dtype", DType_Config)
def test_ConvTranspose(tmpdir, dtype, device_id):
    if device_id == -1 and dtype == np.float16:
        pytest.skip('Test is skipped on CPU with float16 data')
    if dtype == np.float16:
        pytest.skip('Test is temporarily skipped on float16 due to onnxrt bug comparing inf to inf.')
    device = cntk_device(device_id)
    with C.default_options(dtype=dtype):
        # Keep the shapes below as they are, because this tests an earlier bug.
        input_shape = (24, 8, 8) 
        img = np.reshape(np.arange(np.prod(input_shape), dtype = dtype), input_shape) 

        x = C.input_variable(input_shape)

        kernel_shape = (24, 16, 3, 3) # For convolution_transpose the shape is (I x O x W x H)
        kernel = C.constant(value = np.ones(shape=(kernel_shape), dtype = dtype))

        conv_trans_model_with_output_shape = C.convolution_transpose(kernel, x, strides=(2, 2), auto_padding = [False, True, True], output_shape=(16, 16, 16))
        verify_one_input(conv_trans_model_with_output_shape, img, tmpdir, 'ConvTranspose_with_OutputShape_0', device)

        # test without outputShape
        conv_trans_model_without_output_shape = C.convolution_transpose(kernel, x, strides=(2, 2), auto_padding = [False, True, True])
        verify_one_input(conv_trans_model_without_output_shape, img, tmpdir, 'ConvTranspose_without_OutputShape_0', device)

# DepthToSpace
@pytest.mark.parametrize("dtype", DType_Config)
def test_DepthToSpace(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        num_channels = 9
        block_size = 3
        image_shape = (4, 5)
        input_val = np.array(np.reshape(range(num_channels), (num_channels, 1, 1)), dtype=dtype)
        input_val = np.tile(input_val, (1,) + image_shape)
        img = C.input_variable((num_channels,) + image_shape, dtype=dtype)
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

def test_Div_WithBraodcast(tmpdir):
    batch_size, sequence_len = 1, 3
    shape1, shape2 = (1, 2), (2, 1, 1)
    x1 = C.input_variable(shape1, dynamic_axes=[C.Axis.default_batch_axis(), C.Axis('sequenceAxis')])
    x2 = C.input_variable(shape2, dynamic_axes=[])
    model = C.element_divide(x1, x2)
    data1 = np.random.uniform(low=1.0, high=2.0, size=(batch_size, sequence_len, shape1[0], shape1[1])).astype(np.float32)
    data2 = np.random.uniform(low=1.0, high=2.0, size=shape2).astype(np.float32)
    verify_sequence_model(model, [data1, data2], tmpdir, 'test_Div_WithBraodcast', bypass_load_into_cntk = True)

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
    with C.default_options(dtype = dtype):
        c = np.asarray([[0],[1]]).astype(dtype) 
        x = C.input_variable((2,1))
        d = np.arange(12).reshape(6,2).astype(dtype)
        y = C.constant(d)
        x_constant = C.constant(c)
        model = C.gather(y, x_constant)
        verify_no_input(model, tmpdir, 'Gather_0')

        model = C.gather(y, x)
        verify_one_input(model, c, tmpdir, 'Gather_1')

#Gather
@pytest.mark.parametrize("dtype", DType_Config)
def test_Gather_With_Axis(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.asarray( [[ [111, 112], [121, 122], [131, 132], ],[ [211, 212], [221, 222], [231, 232], ]]).astype(dtype)
        indices = np.asarray([[0, 1, 1], [1, 1, 1]])
        y = C.input_variable(np.shape(indices))
        axis = 1

        model = C.gather(data, y, axis, 'gather_with_axis')
        verify_one_input(model, indices, tmpdir, 'Gather_With_Axis_1')

#GlobalAveragePool
@pytest.mark.parametrize("dtype", DType_Config)
def test_GlobalAveragePool(tmpdir, dtype, device_id):
    if device_id == -1 and dtype == np.float16:
        pytest.skip('Test is skipped on CPU with float16 data')
    device = cntk_device(device_id)
    input_shape = [3, 7, 7]
    with C.default_options(dtype=dtype):
        img = np.reshape(np.arange(np.prod(input_shape), dtype = dtype), input_shape)
        x = C.input_variable(img.shape)
        model = C.layers.GlobalAveragePooling()(x)

        verify_one_input(model, img, tmpdir, 'GlobalAveragePool_1', device)

#GlobalMaxPool
@pytest.mark.parametrize("dtype", DType_Config)
def test_GlobalMaxPool(tmpdir, dtype, device_id):
    if device_id == -1 and dtype == np.float16:
        pytest.skip('Test is skipped on CPU with float16 data')
    device = cntk_device(device_id)
    input_shape = [4, 8, 8]
    with C.default_options(dtype=dtype):
        img = np.reshape(np.arange(np.prod(input_shape), dtype = dtype), input_shape)
        x = C.input_variable(img.shape)
        model = C.layers.GlobalMaxPooling()(x)

        verify_one_input(model, img, tmpdir, 'GlobalMaxPool_1', device)        

#Greater
@pytest.mark.parametrize("dtype", DType_Config)
def test_Greater(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        model = C.greater([41., 42., 43.], [42., 42., 42.])
        verify_no_input(model, tmpdir, 'Greater_0')

#GRU
@pytest.mark.parametrize("use_external_files_to_store_parameters", (False, True))
@pytest.mark.parametrize("dtype", DType_Config)
def test_GRU(tmpdir, dtype, use_external_files_to_store_parameters):
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

        input_dim = 200
        cell_dim = 30
        batch_size = 1
        sequence_len = 5

        for config in list(product(direction_options, initial_state_options, activation_options)):
            model_filename = MakeGRUNameFromConfig(*config)
            backward, initial_state, activation =  config
    
            x = C.input_variable(input_dim, dynamic_axes=[C.Axis.default_batch_axis(), C.Axis('sequenceAxis')]) 
            GRUModel = C.layers.Recurrence(C.layers.GRU(cell_dim,     
                                                        activation = activation),   
                                           initial_state = initial_state,    
                                           go_backwards=backward)(x)
            data = np.random.uniform(low=0.0, high=1.0, size=(batch_size, sequence_len, input_dim)).astype(dtype)
            verify_sequence_model(GRUModel, data, tmpdir, model_filename, 
                                  use_external_files_to_store_parameters = use_external_files_to_store_parameters)


#Hardmax
@pytest.mark.parametrize("dtype", DType_Config)
def test_Hardmax(tmpdir, dtype):
    data = np.asarray([1., 1., 2., 3.], dtype)
    model = C.hardmax(data)
    verify_no_input(model, tmpdir, 'Hardmax_0')

    data = np.asarray([[1, 2, 3], [6, 5, 4]], dtype)
    model = C.hardmax(data)
    verify_no_input(model, tmpdir, 'Hardmax_2d_0')


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

        model = C.image_scaler(image, scalar, bias)
        verify_no_input(model, tmpdir, 'ImageScaler_0')

        x = C.input_variable(np.shape(image)) 
        model = C.image_scaler(x, scalar, bias)
        verify_one_input(model, image, tmpdir, 'ImageScaler_1')

#LayerNormalization
@pytest.mark.parametrize("dtype", DType_Config)
def test_LayerNormalization(tmpdir, dtype, device_id):
    if device_id == -1 and dtype == np.float16:
        pytest.skip('Test is skipped on CPU with float16 data')
    # Currently there is a bug on build test. GPU environment is not set correctly for this test. 
    # Thus this test will fail as it will fall back to use CPU with float16.
    if dtype == np.float16:
        pytest.skip('Test is skipped on float16 to pass build test')

    # This test point tests the LayerNormalization round trip with defaultepsilon. We loose always the epsilon value when
    # exporting to ONNX (because ONNX MeanVarianceNormalization does not have an epsilon attribute). When loading back
    # from ONNX, CNTK always uses the default eposilon value (0.00000001). That's why test below has the default epsilon
    # value. It is not expected to pass with any other epsilon value until something changes.
    with C.default_options(dtype = dtype):
        test_shapes = [(3, 5, 7), (10, ), (20, 31)]
        for shape in test_shapes:
            data = np.reshape(np.arange(np.prod(shape), dtype = dtype), shape)
            input_operand = C.input_variable(shape=shape)
            model0 = C.layers.LayerNormalization(initial_scale=1, initial_bias=2, epsilon=0.000000001)(input_operand)
            verify_one_input(model0, data, tmpdir, 'LayerNorm_0' + str(shape).replace(',', '_'), rtol = 1e-04, atol=1e-08)

        # This test point tests especially with epsilon = 0, because that creates a graph with
        # different number of ops. However, we don't expect the numbers to match in round trip
        # because we only support default epislon (0.00000001) when loading from ONNX. Therefore,
        # this is just a load/save test.
        model1 = C.layers.LayerNormalization(epsilon=0.0)(input_operand)
        filename = os.path.join(str(tmpdir), R'LayerNorm_1.onnx')
        model1.save(filename, format=C.ModelFormat.ONNX)
        loaded_model = C.Function.load(filename, format=C.ModelFormat.ONNX)
        model_shape = model1.shape
        if model1.output.dynamic_axes == (C.Axis('defaultBatchAxis'),):
            opname = model1.owner.op_name
            dim_denotation = CNTK_FREEDIM_AXIS_DENOTATION if opname in set_of_batch_ops else DIM_SIZE_FOR_NON_BATCH_OPS
            model_shape = (dim_denotation, ) + model_shape
        assert model_shape == loaded_model.shape

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
    data = np.array([[1, 1, 2, 3]], dtype)
    model = C.log_softmax(data)
    verify_no_input(model, tmpdir, 'LogSoftmax_0')

    x = C.input_variable(data.shape, dtype=dtype)
    model = C.log_softmax(x)
    verify_one_input(model, data, tmpdir, 'LogSoftmax_1')

#LogAddExp
@pytest.mark.parametrize("dtype", DType_Config)
def test_LogAddExp(tmpdir, dtype):
    shape = (2,3,4)

    data_x = np.random.rand(*shape).astype(np.float32)
    data_y = np.random.rand(*shape).astype(np.float32)

    x = C.input_variable(shape)
    y = C.input_variable(shape)

    model = C.log_add_exp(x, y)

    verify_two_input(model, data_x, data_y, tmpdir, 'LogAddExp_0')

@pytest.mark.parametrize("dtype", DType_Config)
def test_LogAddExp_Broadcast(tmpdir, dtype):
    shape_x_arr = [(2,1,4), (2,1,4), (2,2,3,4)]
    shape_y_arr = [(1,3,1), (3,1),   (1,1)]

    for i, (shape_x, shape_y) in enumerate(list(zip(shape_x_arr, shape_y_arr))):
        data_x = np.random.rand(*shape_x).astype(np.float32)
        data_y = np.random.rand(*shape_y).astype(np.float32)

        x = C.input_variable(shape_x)
        y = C.input_variable(shape_y)

        model = C.log_add_exp(x, y)

        verify_two_input(model, data_x, data_y, tmpdir, 'LogAddExp_Broadcast_' + str(i))

#LRN
@pytest.mark.parametrize("dtype", DType_Config)
def test_LRN(tmpdir, dtype, device_id):
    if device_id == -1 and dtype == np.float16:
        pytest.skip('Test is skipped on CPU with float16 data, because it uses convolution.')
    device = cntk_device(device_id)
    with C.default_options(dtype=dtype):
        img_shape = (64, 32, 32)
        img = np.asarray(np.random.uniform(-1, 1, img_shape), dtype=dtype)
        x_r = C.input_variable(shape=img_shape, dtype=dtype)
        model = C.local_response_normalization(x_r, 2, 1.0, 0.0001, 0.75)
        verify_one_input(model, img, tmpdir, 'LRN_1', device)
        # test with edge case kernel size > channel size
        # also test in lotus such that we are getting the value right.
        # in onnx spec and lotus implementation, alpha is divided by size. 
        # so it seems even if size is > and rounded down to channel size,
        # its original value is still used in dividing alpha.
        img_shape = (5, 32, 32)
        img = np.asarray(np.random.uniform(-1, 1, img_shape), dtype=dtype)
        x_r = C.input_variable(shape=img_shape, dtype=dtype)
        model = C.local_response_normalization(x_r, 4, 1.0, 0.0001, 0.75)
        verify_one_input(model, img, tmpdir, 'LRN_2', device)

#LSTM
@pytest.mark.parametrize("use_external_files_to_store_parameters", (False, True))
@pytest.mark.parametrize("dtype", DType_Config)
def test_LSTM(tmpdir, dtype, use_external_files_to_store_parameters):
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

        input_dim = 200
        cell_dim = 30
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
            data = np.random.uniform(low=0.0, high=1.0, size=(batch_size, sequence_len, input_dim)).astype(dtype)
            verify_sequence_model(LSTMmodel, data, tmpdir, model_filename, 
                                  use_external_files_to_store_parameters = use_external_files_to_store_parameters)

#MatMul
@pytest.mark.parametrize("dtype", DType_Config)
def test_MatMul(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data0 = np.asarray([[1,2],[3,4]], dtype=dtype)
        data1 = np.asarray([[5],[6]], dtype=dtype)
        model = C.times(data0, data1)
        verify_no_input(model, tmpdir, 'MatMul_0')

#MatMul 2d
@pytest.mark.parametrize("dtype", DType_Config)
def test_MatMul_2d(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data0 = np.asarray([[1,2],[3,4]], dtype=dtype)
        data1 = np.asarray([[5,7,9],[6,8,10]], dtype=dtype)
        model = C.times(data0, data1)
        verify_no_input(model, tmpdir, 'MatMul_1')

#MatMul 2d with 2 inputs
@pytest.mark.parametrize("dtype", DType_Config)
def test_MatMul_2d_2inputs(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data0 = np.asarray([[1,2],[3,4]], dtype=dtype)
        data1 = np.asarray([[5,7,9],[6,8,10]], dtype=dtype)

        x = C.input_variable(np.shape(data0))
        y = C.input_variable(np.shape(data1))
        model = C.times(x, y)
        verify_two_input(model, data0, data1, tmpdir, 'MatMul_1_1')

#MatMul nd
@pytest.mark.parametrize("dtype", DType_Config)
def test_MatMul_nd(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        np.random.seed(0)

        data0 = np.random.randn(3, 2, 3, 4).astype(np.float32)
        data1 = np.random.randn(2, 3, 4, 5).astype(np.float32)
        model = C.times(data0, data1)
        verify_no_input(model, tmpdir, 'MatMul_n_0')

#MatMul nd
@pytest.mark.parametrize("dtype", DType_Config)
def test_MatMul_nd_2(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        np.random.seed(0)

        data0 = np.random.randn(3, 3, 4).astype(np.float32)
        data1 = np.random.randn(3, 4, 5).astype(np.float32)
        model = C.times(data0, data1)
        verify_no_input(model, tmpdir, 'MatMul_n_1')

#MatMul nd with 2 inputs
@pytest.mark.parametrize("dtype", DType_Config)
def test_MatMul_nd_2inputs(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        np.random.seed(0)

        data0 = np.random.randn(3, 2, 3, 4).astype(np.float32)
        data1 = np.random.randn(2, 3, 4, 5).astype(np.float32)

        x = C.input_variable(np.shape(data0))
        y = C.input_variable(np.shape(data1))
        model = C.times(x, y)
        verify_two_input(model, data0, data1, tmpdir, 'MatMul_n_2')

#MatMul nd with 2 inputs
@pytest.mark.parametrize("dtype", DType_Config)
def test_MatMul_nd_2inputs_2(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        np.random.seed(0)

        data0 = np.random.randn(3, 3, 4).astype(np.float32)
        data1 = np.random.randn(3, 4, 5).astype(np.float32)

        x = C.input_variable(np.shape(data0))
        y = C.input_variable(np.shape(data1))
        model = C.times(x, y)
        verify_two_input(model, data0, data1, tmpdir, 'MatMul_n_3')

@pytest.mark.parametrize("dtype", DType_Config)
def test_CNTK_Times_To_ONNX_MatMul(tmpdir, dtype):
    def generate_matmul_data(input_variable, batch_size, sequence_size):
        np.random.seed(0)
        data_shape = ()
        if input_variable.has_batch_axis():
            data_shape = data_shape + (batch_size,)
        if input_variable.has_sequence_axis():
            data_shape = data_shape + (sequence_size,)
        data_shape = data_shape + input_variable.shape
        data = np.random.standard_normal(data_shape).astype(np.float32)
        return data

    batch_size = 1
    sequence_length = 3
    input1_shape = (2, 3, 4)
    input2_shape = (3, 4, 5, 6)
    output_rank = 2

    ## data_x_data
    x = C.input_variable(input1_shape, dynamic_axes = [])
    y = C.input_variable(input2_shape, dynamic_axes = [])
    model = C.times(x, y, output_rank = output_rank)
    data0 = generate_matmul_data(x, batch_size, sequence_length)
    data1 = generate_matmul_data(y, batch_size, sequence_length)
    verify_two_input(model, data0, data1, tmpdir, 'times_data_x_data')

    ###batch_x_data
    x = C.input_variable(input1_shape, name = "x")
    y = C.input_variable(input2_shape, dynamic_axes = [], name = "y")
    model = C.times(x, y, output_rank = output_rank)
    data0 = generate_matmul_data(x, batch_size, sequence_length)
    data1 = generate_matmul_data(y, batch_size, sequence_length)
    verify_two_input(model, data0, data1, tmpdir, 'batch_x_data')

    ## data_x_batch
    x = C.input_variable(input1_shape, dynamic_axes = [])
    y = C.input_variable(input2_shape)
    model = C.times(x, y, output_rank = output_rank)
    data0 = generate_matmul_data(x, batch_size, sequence_length)
    data1 = generate_matmul_data(y, batch_size, sequence_length)
    verify_two_input(model, data0, data1, tmpdir, 'data_x_batch')

    ## batch_x_batch
    x = C.input_variable(input1_shape)
    y = C.input_variable(input2_shape)
    model = C.times(x, y, output_rank = output_rank)
    data0 = generate_matmul_data(x, batch_size, sequence_length)
    data1 = generate_matmul_data(y, batch_size, sequence_length)
    verify_two_input(model, data0, data1, tmpdir, 'batch_x_batch')

    ### sequence_x_data
    # TODO: ONNX importer cannot handle sequence and batch axes both being free diemention static axis
    #x = C.sequence.input_variable(input1_shape)
    #y = C.input_variable(input2_shape, dynamic_axes = [])
    #model = C.times(x, y, output_rank = output_rank)
    #data0 = generate_matmul_data(x, batch_size, sequence_length)
    #data1 = generate_matmul_data(y, batch_size, sequence_length)
    #verify_sequence_model(model, [data0, data1], tmpdir, 'sequence_x_data')

    ### data_x_sequence
    #TODO: ONNX importer cannot handle sequence and batch axes both being free diemention static axis
    #x = C.input_variable(input1_shape, dynamic_axes = [])
    #y = C.sequence.input_variable(input2_shape)
    #model = C.times(x, y, output_rank = output_rank)
    #data0 = generate_matmul_data(x, batch_size, sequence_length)
    #data1 = generate_matmul_data(y, batch_size, sequence_length)
    #verify_sequence_model(model, [data0, data1], tmpdir, 'data_x_sequence')

    ## sequence_x_sequence
    # TODO: ONNX importer cannot handle sequence and batch axes both being free diemention static axis
    #x = C.sequence.input_variable(input1_shape)
    #y = C.sequence.input_variable(input2_shape)
    #model = C.times(x, y, output_rank = output_rank)
    #data0 = generate_matmul_data(x, batch_size, sequence_length)
    #data1 = generate_matmul_data(y, batch_size, sequence_length)
    #verify_sequence_model(model, [data0, data1], tmpdir, 'sequence_x_sequence')

    ## sequence_x_batch
    # TODO: ONNX importer cannot handle sequence and batch axes both being free diemention static axis
    #x = C.sequence.input_variable(input1_shape)
    #y = C.input_variable(input2_shape)
    #model = C.times(x, y, output_rank = output_rank)
    #data0 = generate_matmul_data(x, batch_size, sequence_length)
    #data1 = generate_matmul_data(y, batch_size, sequence_length)
    #verify_sequence_model(model, [data0, data1], tmpdir, 'sequence_x_batch')

    ## batch_x_sequence
    # TODO: ONNX importer cannot handle sequence and batch axes both being free diemention static axis
    #x = C.input_variable(input1_shape)
    #y = C.sequence.input_variable(input2_shape)
    #model = C.times(x, y, output_rank = output_rank)
    #data0 = generate_matmul_data(x, batch_size, sequence_length)
    #data1 = generate_matmul_data(y, batch_size, sequence_length)
    #verify_sequence_model(model, [data0, data1], tmpdir, 'batch_x_sequence')

#Max
@pytest.mark.parametrize("dtype", DType_Config)
def test_Max(tmpdir, dtype):
    data0 = np.asarray([1., 1., 1., 1.], dtype=dtype)
    data1 = np.asarray([0.5, 0.25, 0.125, 0.], dtype=dtype)
    model = C.element_max(data0, data1)
    verify_no_input(model, tmpdir, 'Max_0')

    data2 = np.asarray([-0.5, 0.26, 0.124, -0.1], dtype=dtype)
    data3 = np.asarray([0.5, -0.26, -0.124, 0.1], dtype=dtype)
    model = C.element_max(data0, data1, data2, data3)
    verify_no_input(model, tmpdir, 'Max_0_4_inputs')

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

        # test for case of not padding but with ceilOutDim=True
        img = np.reshape(np.arange(112*112, dtype=dtype), [1, 112, 112])
        x = C.input_variable(img.shape)
        model = C.pooling(x, C.MAX_POOLING, (3, 3), (2, 2), auto_padding=[False, False, False], ceil_out_dim=True)
        verify_one_input(model, img, tmpdir, 'MaxPool_2', device)

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
    if dtype == np.float16:
        pytest.skip('Mean Variance Normalization with datatype float16 is not supported in ONNX.')
    with C.default_options(dtype = dtype):
        shape = (3, 5, 7)
        data = np.reshape(np.arange(np.prod(shape), dtype = dtype), shape)

        input_operand = C.input_variable(shape=shape)

        model0 = C.mean_variance_normalization(input_operand, use_stats_across_channels=False, do_variance_scaling=True)
        verify_one_input(model0, data, tmpdir, 'MVN_0')

        # do_variance_scaling = False is no longer supported in onnx.
        # model1 = C.mean_variance_normalization(input_operand, use_stats_across_channels=False, do_variance_scaling=False)
        # verify_one_input(model1, data, tmpdir, 'MVN_1')

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
    data0 = np.asarray([1., 1., 1., 1.], dtype=dtype)
    data1 = np.asarray([0.5, 0.25, 0.125, 0.], dtype=dtype)
    model = C.element_min(data0, data1)
    verify_no_input(model, tmpdir, 'Min_0')

    data2 = np.asarray([-0.5, 0.26, 0.124, -0.1], dtype=dtype)
    data3 = np.asarray([0.5, -0.26, -0.124, 0.1], dtype=dtype)
    model = C.element_min(data0, data1, data2, data3)
    verify_no_input(model, tmpdir, 'Min_0_4_inputs')

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
    if device_id == -1:
        pytest.skip('Test only runs on GPU')
    dev = cntk_device(device_id)
    from _cntk_py import constant_initializer
    model_filename = 'optimized_rnn_stack_' + ('bi' if bidirectional else 'uni') + '_layers' + str(num_layers) + '_inp' + str(input_size) + '_hid' + str(hidden_size)
    W = C.parameter((C.InferredDimension, input_size), constant_initializer(0.1), device=dev)
    x = C.sequence.input_variable(shape=(input_size,))
    s = np.asarray(np.random.uniform(-1, 1, (1, 5, input_size)), dtype=np.float32)
    f = C.optimized_rnnstack(x, W, hidden_size, num_layers, bidirectional=bidirectional, recurrent_op=recurrent_op, name='MyRnnStack')
    f.parameters[0].value = np.reshape(np.arange(np.prod(f.parameters[0].value.shape), dtype=np.float32), f.parameters[0].value.shape)
    verify_sequence_model(f, s, tmpdir, model_filename, resave = False)

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
@pytest.mark.parametrize("dtype", DType_Config)
def test_PRelu(tmpdir, dtype):
    # no input
    x_data = np.asarray([[-1, -0.5, 0, 1, 2]], dtype=dtype)
    x = C.constant(value=x_data, dtype=dtype)
    alpha_data = np.asarray([[0.5, 0.5, 0.5, 0.5, 0.5]], dtype=dtype)
    alpha = C.constant(value=alpha_data, dtype=dtype)
    model = C.param_relu(alpha, x)
    verify_no_input(model, tmpdir, 'PRelu_0')

    # one input
    x = C.input_variable(x_data.shape, dtype=dtype)
    model = C.param_relu(alpha, x)
    verify_one_input(model, x_data, tmpdir, 'PRelu_1')

    # two input
    x = C.input_variable(x_data.shape, dtype=dtype)
    alpha = C.input_variable(alpha_data.shape, dtype=dtype)
    model = C.param_relu(alpha, x)
    verify_two_input(model, alpha_data, x_data, tmpdir, 'PRelu_2')

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

        model = C.reduce_l1(data, C.Axis.all_static_axes())
        verify_no_input(model, tmpdir, 'ReduceL1_2')

        x = C.input_variable(data.shape)
        model = C.reduce_l1(x, C.Axis.default_batch_axis())
        verify_one_input(model, data, tmpdir, 'ReduceL1_3')

        model = C.reduce_l1(x, C.Axis.all_axes())
        verify_one_input(model, data, tmpdir, 'ReduceL1_4')

@pytest.mark.parametrize("dtype", DType_Config)
def test_ReduceL2(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.array([[[1,2], [3,4]],[[5,6], [7,8]],[[9,10], [11,12]]], dtype=dtype)
        model = C.reduce_l2(data, 0)
        verify_no_input(model, tmpdir, 'ReduceL2_0')

        model = C.reduce_l2(data, C.Axis.all_static_axes())
        verify_no_input(model, tmpdir, 'ReduceL2_1')

        x = C.input_variable(data.shape)
        model = C.reduce_l2(x, C.Axis.default_batch_axis())
        verify_one_input(model, data, tmpdir, 'ReduceL2_2')

@pytest.mark.parametrize("dtype", DType_Config)
def test_ReduceSumSquare(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.array([[[1,2], [3,4]],[[5,6], [7,8]],[[9,10], [11,12]]], dtype=dtype)
        model = C.reduce_sum_square(data, 0)
        verify_no_input(model, tmpdir, 'ReduceSumSquare_0')

        model = C.reduce_sum_square(data, C.Axis.all_static_axes())
        verify_no_input(model, tmpdir, 'ReduceSumSquare_1')

        x = C.input_variable(data.shape)
        model = C.reduce_sum_square(x, C.Axis.default_batch_axis())
        verify_one_input(model, data, tmpdir, 'ReduceSumSquare_2')

        model = C.reduce_sum_square(x, C.Axis.all_axes())
        verify_one_input(model, data, tmpdir, 'ReduceSumSquare_3')

#ReduceLogSum
@pytest.mark.parametrize("dtype", DType_Config)
def test_ReduceLogSum(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=dtype)
        model = C.reduce_log_sum_exp(data, axis=0)

        verify_no_input(model, tmpdir, 'ReduceLogSum_0')

        model = C.reduce_log_sum_exp(data, C.Axis.all_static_axes())
        verify_no_input(model, tmpdir, 'ReduceLogSum_1')

        x = C.input_variable(data.shape)
        model = C.reduce_log_sum_exp(x, C.Axis.default_batch_axis())
        verify_one_input(model, data, tmpdir, 'ReduceLogSum_2')

#ReduceMax
@pytest.mark.parametrize("dtype", DType_Config)
def test_ReduceMax(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=dtype)
        model = C.reduce_max(data, 0)
        verify_no_input(model, tmpdir, 'ReduceMax_0')

        model = C.reduce_max(data, C.Axis.all_static_axes())
        verify_no_input(model, tmpdir, 'ReduceMax_1')

        x = C.input_variable(data.shape)
        model = C.reduce_max(x, C.Axis.default_batch_axis())
        verify_one_input(model, data, tmpdir, 'ReduceMax_2')

#ReduceMean
@pytest.mark.parametrize("dtype", DType_Config)
def test_ReduceMean(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=dtype)
        model = C.reduce_mean(data, 0)
        verify_no_input(model, tmpdir, 'ReduceMean_0')

        model = C.reduce_mean(data, C.Axis.all_static_axes())
        verify_no_input(model, tmpdir, 'ReduceMean_1')

        x = C.input_variable(data.shape)
        model = C.reduce_mean(x, C.Axis.default_batch_axis())
        verify_one_input(model, data, tmpdir, 'ReduceMean_2')

#ReduceMin
@pytest.mark.parametrize("dtype", DType_Config)
def test_ReduceMin(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=dtype)
        model = C.reduce_min(data, 0)
        verify_no_input(model, tmpdir, 'ReduceMin_0')

        model = C.reduce_min(data, C.Axis.all_static_axes())
        verify_no_input(model, tmpdir, 'ReduceMin_1')

        x = C.input_variable(data.shape)
        model = C.reduce_min(x, C.Axis.default_batch_axis())
        verify_one_input(model, data, tmpdir, 'ReduceMin_2')

#ReduceProd
@pytest.mark.parametrize("dtype", DType_Config)
def test_ReduceProd(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=dtype)
        model = C.reduce_prod(data, 0)
        verify_no_input(model, tmpdir, 'ReduceProd_0')

        model = C.reduce_prod(data, C.Axis.all_static_axes())
        verify_no_input(model, tmpdir, 'ReduceProd_1')

        x = C.input_variable(data.shape)
        model = C.reduce_prod(x, C.Axis.default_batch_axis())
        verify_one_input(model, data, tmpdir, 'ReduceProd_2')

#ReduceSum
@pytest.mark.parametrize("dtype", DType_Config)
def test_ReduceSum(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        data = np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=dtype)
        model = C.reduce_sum(data, 0)
        verify_no_input(model, tmpdir, 'ReduceSum_0')

        model = C.reduce_sum(data, [0, 1, 2])
        verify_no_input(model, tmpdir, 'ReduceSum_1')

        model = C.reduce_sum(data, [0, 2])
        verify_no_input(model, tmpdir, 'ReduceSum_2')

        model = C.reduce_sum(data, [0, 2], keepdims=False)
        verify_no_input(model, tmpdir, 'ReduceSum_3')

        model = C.reduce_sum(data, C.Axis.all_static_axes())
        verify_no_input(model, tmpdir, 'ReduceSum_4')

        x = C.input_variable(data.shape)
        model = C.reduce_sum(x, C.Axis.default_batch_axis())
        verify_one_input(model, data, tmpdir, 'ReduceSum_5')

        model = C.reduce_sum(x, C.Axis.all_axes())
        verify_one_input(model, data, tmpdir, 'ReduceSum_6')

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
        data = np.asarray([[[0., 1.],[2., 3.],[4., 5.]]], dtype)
        i1 = C.input_variable(shape=(3,2))
        model = C.reshape(i1, (2,3))
        verify_one_input(model, data, tmpdir, 'Reshape_1')

#RNN
@pytest.mark.parametrize("use_external_files_to_store_parameters", (False, True))
@pytest.mark.parametrize("dtype", DType_Config)
def test_RNN(tmpdir, dtype, use_external_files_to_store_parameters):
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

        input_dim = 200
        hidden_dim = 30
        batch_size = 1
        sequence_len = 5

        for config in list(product(direction_options, num_layers_options, initial_state_options, activation_options)):
            model_filename = MakeRNNNameFromConfig(*config)
            direction, num_layers, initial_state, activation = config
    
            x = C.input_variable(input_dim, dynamic_axes=[C.Axis.default_batch_axis(), C.Axis('sequenceAxis')]) 
            RNNModel = CreatRNN(
                hidden_dim, 
                activation,  
                initial_state, 
                direction, 
                num_layers)(x)
            data = np.random.uniform(low=0.0, high=1.0, size=(batch_size, sequence_len, input_dim)).astype(dtype)
            verify_sequence_model(RNNModel, data, tmpdir, model_filename, resave = False,
                                  use_external_files_to_store_parameters = use_external_files_to_store_parameters)

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

        model = C.slice(x1, [0,1], [1,0], [2,1])
        verify_one_input(model, data, tmpdir, 'Slice2_1')

        data = np.asarray([[[1,1,1,1],[2,2,2,2],[3,3,2,2]], [[4,4,5,5], [5,5,6,6], [6,6,7,7]]],dtype=dtype)
        x1 = C.input_variable((2,3,4))
        model = C.slice(x1, [1,2], [1,0],[2,1])
        verify_one_input(model, data, tmpdir, 'Slice3_1')

#Sequence.Slice 
@pytest.mark.parametrize("beginIndex, endIndex", (  
    (-2, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-4, 2), (0, 1), (1, 2)))
@pytest.mark.parametrize("dtype", DType_Config)
def test_SequenceSlice(tmpdir, dtype, beginIndex, endIndex):
    with C.default_options(dtype = dtype):
        if dtype == np.float16:
            pytest.skip('Float16 is not supported in CNTK for sequence slice.')
        batch_size = 1
        sequence_length = 5
        input_size = 3
        feature_shape = (input_size,)
        shape = (batch_size, sequence_length, input_size)
        data = np.reshape(range(0, np.prod(shape)), shape).astype(dtype)
        testName = "test_sequence_slice_{0}.{1}".format(beginIndex, endIndex)
        model = C.sequence.slice(C.sequence.input_variable(feature_shape), beginIndex, endIndex)
        verify_sequence_model(model, data, tmpdir, testName)

@pytest.mark.parametrize("dtype", DType_Config)
def test_SequenceFirst(tmpdir, dtype):
    x = C.sequence.input_variable(shape=(3,2))
    y = C.sequence.first(x)
    # create one sequence of 4 tensors each with shape (3,2)
    x0 = np.reshape(np.arange(24.0,dtype=np.float32),(1,4,3,2))
    verify_sequence_model(y, x0, tmpdir, "SequenceFirst")

@pytest.mark.parametrize("dtype", DType_Config)
def test_SequenceLast(tmpdir, dtype):
    x = C.sequence.input_variable(shape=(3,2))
    y = C.sequence.last(x)
    # create one sequence of 4 tensors each with shape (3,2)
    x0 = np.reshape(np.arange(24.0,dtype=np.float32),(1,4,3,2))
    verify_sequence_model(y, x0, tmpdir, "SequenceLast")

def test_SequenceIsFirst(tmpdir):
    batch_size = 1
    sequence_length = 4
    input_shape = (3,2)
    shape = (batch_size, sequence_length, 3, 2)
    data = np.reshape(range(0, np.prod(shape)), shape).astype(np.float32)
    model = C.sequence.is_first(C.sequence.input_variable(input_shape))
    verify_sequence_model(model, data, tmpdir, 'SequenceIsFirst', bypass_load_into_cntk = True)
    
def test_SequenceIsLast(tmpdir):
    batch_size = 1
    sequence_length = 5
    input_shape = (4,3)
    shape = (batch_size, sequence_length, 4, 3)
    data = np.reshape(range(0, np.prod(shape)), shape).astype(np.float32)
    model = C.sequence.is_last(C.sequence.input_variable(input_shape))
    verify_sequence_model(model, data, tmpdir, 'SequenceIsLast', bypass_load_into_cntk = True)
    
@pytest.mark.parametrize("dtype", DType_Config)
def test_SequenceReduceSum(tmpdir, dtype):
    x = C.sequence.input_variable(shape=(3,2))
    # create one sequence of 4 tensors each with shape (3,2)
    x0 = np.reshape(np.arange(24.0,dtype=np.float32),(1,4,3,2))
    y = C.sequence.reduce_sum(x)
    #y.eval({x:x0})
    verify_sequence_model(y, x0, tmpdir, "SequenceReduceSum")

@pytest.mark.parametrize("dtype", DType_Config)
def test_SequenceReduceMax(tmpdir, dtype):
    x = C.sequence.input_variable(shape=(3,2))
    # create one sequence of 4 tensors each with shape (3,2)
    x0 = np.reshape(np.arange(24.0,dtype=np.float32),(1,4,3,2))
    y = C.sequence.reduce_max(x)
    #y.eval({x:x0})
    verify_sequence_model(y, x0, tmpdir, "SequenceReduceMax")

@pytest.mark.parametrize("dtype", DType_Config)
def test_SequenceSoftmax(tmpdir, dtype):
    if dtype==np.float16:
        pytest.skip('Test is skipped with float16 data. Implementation of sequence.softmax is not numerically stable.')
    batch_size, sequence_length, input_size = 1, 2, 1
    a = np.array([[[1],[0]]], dtype)
    src = C.sequence.input_variable(shape=(input_size), sequence_axis=C.Axis("Seq"), dtype=dtype)
    out = C.sequence.softmax(src)
    verify_sequence_model(out, a, tmpdir, "SequenceSoftmax")

#Softmax
@pytest.mark.parametrize("dtype", DType_Config)
def test_Softmax(tmpdir, dtype):
    data = np.array([[1, 1, 2, 3]], dtype)
    model = C.softmax(data)
    verify_no_input(model, tmpdir, 'Softmax_0')

    x = C.input_variable(data.shape, dtype=dtype)
    model = C.softmax(x)
    verify_one_input(model, data, tmpdir, 'Softmax_1')

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
def test_Squeeze(tmpdir):
    pytest.skip('TODO: need to bump ONNX CI version. ')
    x0 = np.arange(6).reshape((1, 2, 1, 3)).astype('f')
    x = C.input_variable((2, 1, 3))
    model = C.squeeze(x, [1])
    verify_one_input(model, x0, tmpdir, 'Squeeze_0')

def test_Squeeze_without_axes(tmpdir):
    pytest.skip('ONNX should update attribute axes to be optional.')
    x0 = np.arange(6).reshape((1, 2, 1, 3)).astype('f')
    x = C.input_variable((2, 1, 3))
    model = C.squeeze(x)
    verify_one_input(model, x0, tmpdir, 'Squeeze_without_axes_0')

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

        model = C.sum([in1])
        verify_one_input(model, in1_data, tmpdir, 'Sum_1')

        model = C.sum([in1, in2, in1])
        verify_two_input(model, in1_data, in2_data, tmpdir, 'Sum_3')

# SpaceToDepth
@pytest.mark.parametrize("dtype", DType_Config)
def test_SpaceToDepth(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        num_channels = 3
        block_size = 3
        image_shape = (12, 15)
        input_val = np.array(np.reshape(range(num_channels), (num_channels, 1, 1)), dtype=dtype)
        input_val = np.tile(input_val, (1,) + image_shape)
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

#TopK
@pytest.mark.parametrize("dtype", DType_Config)
def test_TopK(tmpdir, dtype):
    if dtype == np.float16:
        pytest.skip("TopK of float16 not supported in cntk: Unsupported template argument(half) in SortPairsDescending.")
    input_size = 9
    data = (np.arange(input_size,dtype=dtype)*0.1 + 0.1).reshape(input_size)
    x = C.input_variable(input_size, dtype=dtype)
    model = C.top_k(-x * C.log(x), 3)
    verify_one_input(model, data, tmpdir, "top_k")

#TimesTranspose
@pytest.mark.parametrize("dtype", DType_Config)
def test_TimesTranspose(tmpdir, dtype):
    with C.default_options(dtype = dtype):
        np.random.seed(1)
        data0 = np.random.rand(3, 4).astype(dtype)
        input0 = C.input_variable(data0.shape, dtype = data0.dtype)

        data1 = np.random.rand(4).astype(dtype)
        input1 = C.input_variable(data1.shape, dtype = data1.dtype)
        model = C.times_transpose(input0, input1)
        verify_two_input(model, data0, data1, tmpdir, 'TimesTranspose_0')

        data1 = np.random.rand(5, 4).astype(dtype)
        input1 = C.input_variable(data1.shape, dtype = data1.dtype)
        model = C.times_transpose(input0, input1)
        verify_two_input(model, data0, data1, tmpdir, 'TimesTranspose_1')

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

# Crop
@pytest.mark.parametrize("dtype", DType_Config)
def test_Crop_Manual(tmpdir, dtype):
    x = C.input_variable((1,4,4), dtype=np.float32, name='feature')
    y = C.constant(np.ones((1,2,1), dtype=np.float32))
    model = C.crop_manual(x, y, 1, 2, name='crop_manual')
    data = np.asarray(range(4*4), dtype=np.float32).reshape((1,4,4))
    verify_one_input(model, data, tmpdir, "Crop_Manual_0")

# eye_like
@pytest.mark.parametrize("dtype", DType_Config)
def test_Eye_Like(tmpdir, dtype):
    x = C.input_variable((4, 4), dynamic_axes=[], dtype=dtype, name='feature')
    model = C.eye_like(x, sparse_output=False)
    data = np.asarray(range(4*4), dtype=dtype).reshape((4,4))
    verify_one_input(model, data, tmpdir, "Eye_Like_0")

# zeros_like
@pytest.mark.parametrize("dtype", DType_Config)
def test_Zeros_Like(tmpdir, dtype):
    x = C.input_variable((3, 4), dynamic_axes=[], dtype=dtype, name='feature')
    model = C.zeros_like(x, name='zeros_like_op')
    data = np.asarray(range(3*4), dtype=dtype).reshape((3,4))
    # TODO: import not yet implemented.
    verify_one_input(model, data, tmpdir, "Zeros_Like_0", bypass_load_into_cntk=True)

# ones_like
@pytest.mark.parametrize("dtype", DType_Config)
def test_Ones_Like(tmpdir, dtype):
    x = C.input_variable((3, 4), dynamic_axes=[], dtype=dtype, name='feature')
    model = C.ones_like(x, name='ones_like_op')
    data = np.asarray(range(3*4), dtype=dtype).reshape((3,4))
    # TODO: import not yet implemented.
    verify_one_input(model, data, tmpdir, "Ones_Like_0", bypass_load_into_cntk=True)

# one hot
@pytest.mark.parametrize("dtype", DType_Config)
def test_One_Hot(tmpdir, dtype):
    if dtype == np.float16:
        pytest.skip('float16 not supported in onnxruntime.')
    data = np.asarray([1, 5], dtype=dtype)
    x = C.input_variable((2), dtype=dtype)
    model = C.one_hot(x, 6, False, name='one_hot_op')
    verify_one_input(model, data, tmpdir, "One_Hot_0", bypass_load_into_cntk=True)

    model = C.one_hot(x, 6, False, axis = 0, name='one_hot_op')
    verify_one_input(model, data, tmpdir, "One_Hot_1", bypass_load_into_cntk=True)