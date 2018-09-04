# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import numpy as np
import cntk as C
import pytest
onnx = pytest.importorskip("onnx")

# check whether the input data is a batch of sparse matrices 
def is_list_of_sparse(data):
    return type(data)==list and type(data[0])==scipy.sparse.csr.csr_matrix

# convert a list of sparse matrices to a dense matrix to be used in ONNX test cases 
def sparse_to_dense(sparse_data):
    dense_data = [sparse_data[0].todense()]
    for i in range(1, len(dense_data)):
        dense_data = np.concatenate(dense_data, sparse_data[i].todense()) 
    return dense_data

# ONNX models and CNTK models imported from ONNX are different from original CNTK models in that 
# their inputs take data in the form of [sequence, batch, *feature] (as oppose to [batch, sequence, *feature]).
# this function transposes input data so that it can be used to test ONNX models and imported CNTK models.
def transpose_dynamic_axis(data):
    rank = data.ndim
    assert rank >= 2
    perm = np.arange(rank)
    perm[0], perm[1] = perm[1], perm[0]
    return np.transpose(data, perm)

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
    if variable.has_sequence_axis(): # and variable.is_input:
        data = transpose_dynamic_axis(data)

    tp = onnx.TensorProto()
    tp.name = variable.uid

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
        filename = os.path.join(str(test_model_path), name + R'.onnx')
        model.save(filename, format=C.ModelFormat.ONNX)
        loaded_model = C.Function.load(filename, format=C.ModelFormat.ONNX)
        onnx_model = onnx.load(filename);

        filename_resave = os.path.join(str(test_model_path), name + R'_resave.onnx')
        loaded_model.save(filename_resave, format=C.ModelFormat.ONNX)
        
    return loaded_model, onnx_model, test_model_path, test_data_path

def save_test_data(model, onnx_model, test_data_path, input_data, output_data, name, tmpdir):
    if not onnx_model:
        return;

    if (len(model.arguments) == 1):
        save_cntk_data_as_onnx_tensor(os.path.join(str(test_data_path), 'input_{0}.pb'.format(0)), 
                        model.arguments[0], input_data, onnx_model.graph.input[0]) #, data_type = np.int)
    else:
        for i in range(len(model.arguments)):
            save_cntk_data_as_onnx_tensor(os.path.join(str(test_data_path), 'input_{0}.pb'.format(i)), 
                            model.arguments[i], input_data[i], onnx_model.graph.input[i])

    if (len(model.outputs) > 1):
        for i in range(0, len(model.outputs)): 
            output_data_i = output_data[model.outputs[i]]
            save_cntk_data_as_onnx_tensor(os.path.join(str(test_data_path), 'output_{0}.pb'.format(i)), 
                            model.outputs[i], output_data_i, onnx_model.graph.output[i])
    else:
        save_cntk_data_as_onnx_tensor(os.path.join(str(test_data_path), 'output_{0}.pb'.format(0)), 
                        model.outputs[0], output_data, onnx_model.graph.output[0])

    # print out command line for onnx test runner
    print(R'onnx_test_runner.exe -n ' + name + ' ' + str(tmpdir))
