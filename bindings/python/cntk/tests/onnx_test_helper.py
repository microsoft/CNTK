# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import numpy as np
import scipy
import cntk as C
import pytest
onnx = pytest.importorskip("onnx")

CNTK_FREEDIM_AXIS_DENOTATION = -3
DIM_SIZE_FOR_NON_BATCH_OPS = 1

# check whether the input data is a batch of sparse matrices 
def is_list_of_sparse(data):
    return type(data)==list and type(data[0])==scipy.sparse.csr.csr_matrix

# convert a list of sparse matrices to a dense matrix to be used in ONNX test cases 
def sparse_to_dense(sparse_data):
    dense_data = [sparse_data[0].todense()]
    for i in range(1, len(dense_data)):
        dense_data = np.concatenate(dense_data, sparse_data[i].todense()) 
    return np.array(dense_data)

# ONNX models and CNTK models imported from ONNX are different from original CNTK models in that 
# their inputs take data in the form of [sequence, batch, *feature] (as oppose to [batch, sequence, *feature]).
# this function transposes input data so that it can be used to test ONNX models and imported CNTK models.
def transpose_dynamic_axis(data):
    rank = data.ndim
    assert rank >= 2
    perm = np.arange(rank)
    perm[0], perm[1] = perm[1], perm[0]
    return np.transpose(data, perm)

# find index to the sequence axis in a ONNX tensor that would be converted from a CNTK variable.
def get_sequence_axis_index(output_variable):
    for i in range(0, len(output_variable.dynamic_axes)):
        axis = output_variable.dynamic_axes[i]
        if axis.is_sequence_axis:
            return i
    for i in range(0, len(output_variable.shape)):
        if output_variable.shape[i] == CNTK_FREEDIM_AXIS_DENOTATION:
            return i + len(output_variable.dynamic_axes)
    return -1;

# check whether two CNTK output variables have sequence axis at different dimensions. 
# it indicates that data outputs need to be transposed before comparison.
# it is safe to assume that sequence dimension is either 0 or 1.
def compare_model_for_output_data_transpose(model_output, loaded_model_output):
    model_sequence_index = get_sequence_axis_index(model_output)
    loaded_model_sequence_index = get_sequence_axis_index(loaded_model_output)

    return model_sequence_index != -1 and loaded_model_sequence_index != -1 and model_sequence_index != loaded_model_sequence_index

# find index to the sequence axis in an ONNX tensor
def get_onnx_free_dimension_index(onnx_value_info_proto):
    indices = [onnx_free_dim_index for onnx_free_dim_index, d in enumerate(onnx_value_info_proto.type.tensor_type.shape.dim) if d.dim_param == "Sequence"]
    if len(indices) != 1:
        return -1;
    return indices[0]

# check whether a CNTK variable and a ONNX ValueInfoProto have sequence axis at different dimensions. 
# it indicates that data outputs need to be transposed before comparison.
# it is safe to assume that sequence dimension is either 0 or 1.
def compare_output_for_data_transpose(variable, onnx_value_info_proto):
    model_sequence_index = get_sequence_axis_index(variable)
    loaded_model_sequence_index = get_onnx_free_dimension_index(onnx_value_info_proto)

    return model_sequence_index != -1 and loaded_model_sequence_index != -1 and model_sequence_index != loaded_model_sequence_index

# Save numpy data used for CNTK model in ONNX tensor format. The followings are handled in the function.
# CNTK data is usually float. It can be sparse or dense.
# ONNX tensor data type depends on its ValueInfoProto attribute. ONNX does not support sparse densors.
# Depending on the use case, data may need to be transposed to be used with ONNX models.
def save_cntk_data_as_onnx_tensor(file_path, variable, data, onnx_value_info_proto):
    # sequence mode data shape: (batch, sequecen, ...) 
    # to make onnx model work, batch must be 1
    # swith to onnx shape: (sequence, batch, ...)
    if is_list_of_sparse(data):
        data = sparse_to_dense(data)

    # compare free_dim indices between variable with onnx_value_info_proto
    # they are at index 0 and 1. 
    if compare_output_for_data_transpose(variable, onnx_value_info_proto):
        data = transpose_dynamic_axis(data)

    tp = onnx.TensorProto()
    tp.name = onnx_value_info_proto.name

    shape = np.shape(data)
    for i in range(0, len(shape)):
        tp.dims.append(shape[i])

    if type(data) == list:
        # this is probably because of batch (list of samples)
        data = data[0]

    tp.data_type = onnx_value_info_proto.type.tensor_type.elem_type
    if onnx_value_info_proto.type.tensor_type.elem_type == onnx.TensorProto.DOUBLE:
        data=data.astype(np.double)
        tp.raw_data = data.tobytes()
    elif onnx_value_info_proto.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
        tp.raw_data = data.tobytes()
    elif onnx_value_info_proto.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
        tp.raw_data = data.tobytes()
    elif onnx_value_info_proto.type.tensor_type.elem_type == onnx.TensorProto.INT64:
        data=data.astype(np.int64)
        tp.raw_data = data.tobytes()
    elif onnx_value_info_proto.type.tensor_type.elem_type == onnx.TensorProto.INT32:
        data=data.astype(np.int)
        tp.raw_data = data.tobytes()
    elif onnx_value_info_proto.type.tensor_type.elem_type == onnx.TensorProto.BOOL:
        data=data.astype(np.bool)
        tp.raw_data = data.tobytes()
    else:
        assert False, R'Tensor element type not supported: ' + onnx.TensorProto.DataType.Name(onnx_value_info_proto.type.tensor_type.elem_type)
        
    with open(file_path, 'wb') as f:
        f.write(tp.SerializeToString())

#
# This function creates and populates a folder structure for an ONNX test case. 
# To test repeated conversion, a CNTK model is converted to ONNX, converted back to CNTK, and then reconverted to ONNX. 
# This reconverted model is saved along with the ONNX test case data. 
# Test data are put into the folder later after model evaluation. 
#
# Folder structure for ONNX test cases is like the following:
# tmpdir (folder) 
#   test_named_test_case0 (folder)
#       test_data_set_0 (folder)
#           input_0.pb
#           output_0.pb
#       onnx_model    
#   resave_test_model_for_test_case0
#   test_named_test_case1 (folder)
#       test_data_set_0 (folder)
#           input_0.pb
#           output_0.pb
#       onnx_model    
#   resave_test_model_for_test_case1
#       
def create_and_populate_onnx_test_case_with_model_conversion(model, tmpdir, name, loaded_model):
    onnx_model = None
    test_model_path = os.path.join(str(tmpdir), R'test_' + name)
    os.mkdir(test_model_path)
    test_data_path = os.path.join(str(test_model_path), R'test_data_set_0')
    os.mkdir(test_data_path)
    if not loaded_model:
        ## leave this line for debugging when needed
        ## plot original model
        #C.logging.graph.plot(model, os.path.join(str(test_model_path), name + ".pdf"))

        filename = os.path.join(str(test_model_path), name + R'.onnx')
        model.save(filename, format=C.ModelFormat.ONNX)

        loaded_model = C.Function.load(filename, format=C.ModelFormat.ONNX)
        onnx_model = onnx.load(filename);

        ## leave this line for debugging when needed
        # plot loaded model
        #C.logging.graph.plot(loaded_model, filename + ".pdf")

        filename_resave = os.path.join(str(test_model_path), name + R'_resave.onnx')
        loaded_model.save(filename_resave, format=C.ModelFormat.ONNX)
        
    return loaded_model, onnx_model, test_model_path, test_data_path

# onnx model outputs are not necessarily in the same order as the original CNTK model.
# it may also have additional outputs (those not being combined as output in a CNTK model).
# when exporting a CNTK model, variable uid is used to name an onnx node arg. 
# however, for some outputs, we have to extent it with a noop so it can be treated as onnx output.
# in such case, the onnx output will have a name with uid as prefix (e.g. "Reshape3635_Output_0" + "_attach_noop_")
# this funcion is to find an onnx output based on a CNTK variable uid according to above naming scheme.
def find_onnx_value_info_proto_with_matching_name(onnx_outputs, cntk_output_uid, fallback_onnx_output):
    for i in range(0, len(onnx_outputs)):
        onnx_output_name = onnx_outputs[i].name
        if onnx_output_name == cntk_output_uid:
            return onnx_outputs[i]

    # not able to find exact match. find a close one.
    for i in range(0, len(onnx_outputs)):
        onnx_output_name = onnx_outputs[i].name
        if onnx_output_name.find(cntk_output_uid) == 0:
            return onnx_outputs[i]

    return fallback_onnx_output

def save_test_data(model, onnx_model, test_data_path, input_data, output_data, name, tmpdir):
    if not onnx_model:
        return;

    if (len(model.arguments) == 1):
        onnx_value_info_proto = find_onnx_value_info_proto_with_matching_name(
            onnx_model.graph.input, model.arguments[0].uid, onnx_model.graph.input[0])
        save_cntk_data_as_onnx_tensor(os.path.join(str(test_data_path), 'input_{0}.pb'.format(0)), 
                        model.arguments[0], input_data, onnx_value_info_proto) #, data_type = np.int)
    else:
        for i in range(len(model.arguments)):
            onnx_value_info_proto = find_onnx_value_info_proto_with_matching_name(
                onnx_model.graph.input, model.arguments[i].uid, onnx_model.graph.input[i])
            save_cntk_data_as_onnx_tensor(os.path.join(str(test_data_path), 'input_{0}.pb'.format(i)), 
                            model.arguments[i], input_data[i], onnx_value_info_proto)

    if (len(model.outputs) == 1):
        onnx_value_info_proto = find_onnx_value_info_proto_with_matching_name(
            onnx_model.graph.output, model.outputs[0].uid, onnx_model.graph.output[0])
        save_cntk_data_as_onnx_tensor(os.path.join(str(test_data_path), 'output_{0}.pb'.format(0)), 
                        model.outputs[0], output_data, onnx_value_info_proto)
    else:
        for i in range(0, len(model.outputs)): 
            output_data_i = output_data[model.outputs[i]]
            onnx_value_info_proto = find_onnx_value_info_proto_with_matching_name(
                onnx_model.graph.output, model.outputs[i].uid, onnx_model.graph.output[i])
            save_cntk_data_as_onnx_tensor(os.path.join(str(test_data_path), 'output_{0}.pb'.format(i)), 
                            model.outputs[i], output_data_i, onnx_value_info_proto)

    # print out command line for onnx test runner
    print(R'onnx_test_runner.exe -n ' + name + ' ' + str(tmpdir))
