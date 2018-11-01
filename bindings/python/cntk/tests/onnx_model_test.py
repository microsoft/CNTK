# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import cntk as C
import numpy as np
import pytest
import os
import re
import shutil
import time
import tempfile

onnx = pytest.importorskip("onnx")
from onnx import numpy_helper

from .onnx_test_helper import find_onnx_value_info_proto_with_matching_name, save_cntk_data_as_onnx_tensor
from .onnx_verify_helper import verify_model

# To test models locally, create folder 'onnx_models' and put in model folders. 
# For example.
#   .
#   +-- onnx_models  # models stored in 'model.onnx' onnx format.
#   |   +-- model1
#   |   |   +-- model.onnx
#   |   |   +-- test_data_set_0
#   |   |   |   +-- input_0.pb
#   |   |   |   +-- input_1.pb
#   |   |   |   +-- output_0.pb
#   |   |   +-- test_data_set_1
#   |   |   |   +-- input_0.pb
#   |   |   |   +-- input_1.pb
#   |   |   |   +-- output_0.pb
#   |   +-- model2
#     ...
#   +-- PretrainedModelsV2  # models stored in '.model' CNTKv2 format.
#   |   +-- model1.model
#   |   +-- model2.model
#     ...
def get_base_dir(base_dir):
    return base_dir if not 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY' in os.environ else os.path.join(os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'], base_dir)
onnx_base_dir = get_base_dir('onnx_models')
onnx_model_names = [dir for dir in os.listdir(onnx_base_dir)
                    if os.path.isdir(os.path.join(onnx_base_dir, dir)) and os.path.exists(os.path.join(onnx_base_dir, dir, 'model.onnx'))] if os.path.exists(onnx_base_dir) else []
cntk_base_dir = get_base_dir('PretrainedModelsV2')
cntk_model_names = [dir for dir in os.listdir(cntk_base_dir)
                    if os.path.isfile(os.path.join(cntk_base_dir, dir)) and dir.rfind('.model') + len('.model') == len(dir)] if os.path.exists(cntk_base_dir) else []
input_filename_pattern = re.compile('input_[0-9]+.pb')
output_filename_pattern = re.compile('output_[0-9]+.pb')

skip_model_names = [
    # Convolution Nan issue on Linux. 
    'shufflenet',
    # Tests from onnx backend tests that currently fails. 
    'test_constant',
    'test_edge_pad',
    'test_gru_defaults',
    'test_gru_seq_length',
    'test_gru_with_initial_bias',
    'test_lstm_defaults',
    'test_lstm_with_initial_bias',
    'test_lstm_with_peepholes',
    'test_reduce_log_sum',
    'test_reduce_log_sum_asc_axes',
    'test_reduce_log_sum_default',
    'test_reduce_log_sum_desc_axes',
    'test_reshape_extended_dims',
    'test_reshape_negative_dim',
    'test_reshape_one_dim',
    'test_reshape_reduced_dims',
    'test_reshape_reordered_dims',
    'test_rnn_seq_length',
    'test_shape',
    'test_shape_example',
    'test_simple_rnn_defaults',
    'test_simple_rnn_with_initial_bias',
    'test_size',
    'test_size_example',
    'test_slice_end_out_of_bounds',
    'test_slice_start_out_of_bounds',
    'test_split_equal_parts_1d',
    'test_split_equal_parts_2d',
    'test_split_equal_parts_default_axis',
    'test_split_variable_parts_1d',
    'test_split_variable_parts_2d',
    'test_split_variable_parts_default_axis',
    'test_sum_one_input',
    'test_thresholdedrelu',
    'test_thresholdedrelu_default',
    'test_thresholdedrelu_example',
    'test_tile',
    'test_tile_precomputed',
    'test_top_k',
    'test_transpose_default',
    'test_upsample_nearest',
]

skip_round_trip_model_names = [
    # Convolution Nan issue on Linux. 
    'shufflenet',
    # Tests from onnx backend tests that currently fails. 
    'test_constant',
    'test_edge_pad',
    'test_gru_defaults',
    'test_gru_seq_length',
    'test_gru_with_initial_bias',
    'test_lstm_defaults',
    'test_lstm_with_initial_bias',
    'test_lstm_with_peepholes',
    'test_reduce_log_sum',
    'test_reduce_log_sum_asc_axes',
    'test_reduce_log_sum_default',
    'test_reduce_log_sum_desc_axes',
    'test_reshape_extended_dims',
    'test_reshape_negative_dim',
    'test_reshape_one_dim',
    'test_reshape_reduced_dims',
    'test_reshape_reordered_dims',
    'test_rnn_seq_length',
    'test_shape',
    'test_shape_example',
    'test_simple_rnn_defaults',
    'test_simple_rnn_with_initial_bias',
    'test_size',
    'test_size_example',
    'test_slice',
    'test_slice_default_axes',
    'test_slice_end_out_of_bounds',
    'test_slice_start_out_of_bounds',
    'test_split_equal_parts_1d',
    'test_split_equal_parts_2d',
    'test_split_equal_parts_default_axis',
    'test_split_variable_parts_1d',
    'test_split_variable_parts_2d',
    'test_split_variable_parts_default_axis',
    'test_sum_one_input',
    'test_thresholdedrelu',
    'test_thresholdedrelu_default',
    'test_thresholdedrelu_example',
    'test_tile',
    'test_tile_precomputed',
    'test_top_k',
    'test_transpose_default',
    'test_upsample_nearest',
]

skip_cntk_model_names = []

@pytest.mark.parametrize('model_name, round_trip',
    [(model_name, round_trip) for model_name in onnx_model_names for round_trip in [False, True]],
    ids=['round_trip_' + model_name if round_trip else model_name for model_name in onnx_model_names for round_trip in [False, True]])
def test_onnx_model(model_name, round_trip):
    if model_name in skip_model_names and not round_trip:
        pytest.skip('Skip onnx model test. ')
    if model_name in skip_round_trip_model_names and round_trip:
        pytest.skip('Skip onnx model round trip test. ')

    model_dir = os.path.join(onnx_base_dir, model_name)
    model = C.Function.load(os.path.join(model_dir, 'model.onnx'), format=C.ModelFormat.ONNX)

    if round_trip:
        resave_model_path = 'model_resave.onnx'
        model.save(resave_model_path, format=C.ModelFormat.ONNX)
        model = C.Function.load(resave_model_path, format=C.ModelFormat.ONNX)

    data_dirs = [os.path.join(model_dir, dir) for dir in os.listdir(model_dir)
                 if os.path.isdir(os.path.join(model_dir, dir))]
    for data_dir in data_dirs:
        inputs = []
        ref_outputs = []
        tensor = onnx.TensorProto()

        input_filenames = [filename for filename in os.listdir(data_dir) if input_filename_pattern.match(filename)]
        input_files_sorted = [os.path.join(data_dir, 'input_{:d}.pb'.format(i)) 
                              for i in range(len(input_filenames))]
        output_filenames = [filename for filename in os.listdir(data_dir) if output_filename_pattern.match(filename)]
        output_files_sorted = [os.path.join(data_dir, 'output_{:d}.pb'.format(i)) 
                               for i in range(len(output_filenames))]

        for input_file in input_files_sorted:
            with open(input_file, 'rb') as f:
                tensor.ParseFromString(f.read())
            inputs.append(numpy_helper.to_array(tensor))

        for output_file in output_files_sorted:
            with open(output_file, 'rb') as f:
                tensor.ParseFromString(f.read())
            ref_outputs.append(numpy_helper.to_array(tensor))

        cntk_input = {model.arguments[i]:inputs[i] for i in range(len(inputs))}
        cntk_res = [model.eval(cntk_input)]

        if ref_outputs[0].dtype == np.bool:
            cntk_res = [cntk_res[0].astype("bool")]

        outputs = list(cntk_res)

        np.testing.assert_equal(len(ref_outputs), len(outputs))
        for i in range(len(outputs)):
            np.testing.assert_equal(ref_outputs[i].dtype, outputs[i].dtype)
            np.testing.assert_allclose(
                ref_outputs[i],
                outputs[i],
                rtol=1e-3,
                atol=1e-4)

# Helper for exporting test data.
model_file = 'model.onnx'
data_dir = 'test_data_set_0'
def SaveData(test_data_dir, prefix, onnx_variables, variables, data_list, names, batch_size=1):
    if isinstance(data_list, np.ndarray):
        data_list = [data_list]
    for (i, d), v, n in zip(enumerate(data_list), variables, names):
        onnx_value_info_proto = find_onnx_value_info_proto_with_matching_name(onnx_variables, n, onnx_variables[0])
        save_cntk_data_as_onnx_tensor(os.path.join(test_data_dir, '{0}_{1}.pb'.format(prefix, i)), v, d, onnx_value_info_proto)

def Save(dir, func, inputs, outputs, batch_size=1):
    if not os.path.exists(dir):
        os.makedirs(dir)
    model_file_path = os.path.join(dir, model_file)
    func.save(model_file_path, C.ModelFormat.ONNX)
    onnx_model = onnx.load(model_file_path)
    onnx_model_description = onnx_model.graph.doc_string
    uid_name_map = dict(tuple(x[3:-3].split(', ')) for x in re.findall(r'<<<[^>]*>>>', onnx_model_description)[1:])
    input_names = [uid_name_map[x.uid] for x in func.arguments]
    # handle block outputs
    output_names = []
    block_uid_count = {}
    # when block are exported as a single onnx node, the onnx node output takes name from block node output.
    # when block are exported by exporting nodes within that block, the onnx node output takes name from inner node output.
    # the cntk node that provides the name will have its uid stored in the uid_name_map.
    # this function tries to find the deepest inner output node whose uid is in uid_name_map.
    def find_deepest_inner_block_output(output):
        # might be a placeholder
        if not output.is_output:
            return False, output
        if output.owner and output.owner.is_block:
            block_uid_count[output.owner.uid] = block_uid_count[output.owner.uid] + 1 if output.owner.uid in block_uid_count else 0
            found, inner_output = find_deepest_inner_block_output(output.owner.block_root.outputs[block_uid_count[output.owner.uid]])
            if found:
                return True, inner_output
        return output.uid in uid_name_map, output

    for output in func.outputs:
        _, output = find_deepest_inner_block_output(output)
        output_names.append(uid_name_map[output.uid])

    test_data_dir = os.path.join(dir, data_dir)
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    SaveData(test_data_dir, 'input', onnx_model.graph.input, func.arguments, inputs, input_names, batch_size)
    SaveData(test_data_dir, 'output', onnx_model.graph.output, func.outputs, outputs, output_names, batch_size)

# Initialize tmp-directory for exporting cntk models
tmpdir = 'tmp_exported_models'
if os.path.isdir(tmpdir):
    # os.mkdir might get called before shutil.rmtree complete. So rename the current tmpdir to avoid collision.
    tmp = tempfile.mktemp(dir=os.path.dirname(tmpdir))
    shutil.move(tmpdir, tmp)
    shutil.rmtree(tmp)
os.mkdir(tmpdir)

# test_cntk_model will create exported onnx model with test data in the following tmp folder:
#   .
#   +-- tmp_exported_models  # models exported in 'model.onnx' onnx format.
#   |   +-- test_model1
#   |   |   +-- model.onnx
#   |   |   +-- test_data_set_0
#   |   |   |   +-- input_0.pb
#   |   |   |   +-- input_1.pb
#   |   |   |   +-- output_0.pb
#   |   |   +-- test_data_set_1
#   |   |   |   +-- input_0.pb
#   |   |   |   +-- input_1.pb
#   |   |   |   +-- output_0.pb
#   |   +-- test_model2
#     ...
@pytest.mark.parametrize('model_name',
    [model_name for model_name in cntk_model_names],
    ids=[model_name for model_name in cntk_model_names])
def test_cntk_model(model_name):
    if model_name in skip_cntk_model_names:
        pytest.skip('Skip cntk model test. ')
    model_dir = os.path.join(cntk_base_dir, model_name)
    model = C.Function.load(model_dir, format=C.ModelFormat.CNTKv2)

    resave_model_dir = os.path.join(tmpdir, 'test_' + model_name)
    resave_model_path = os.path.join(resave_model_dir, model_file)

    np.random.seed(3)
    input_shape = (1,) + model.arguments[0].shape
    data_x = np.asarray(np.random.uniform(-1, 1, input_shape), dtype=np.float32)
    data_y = model.eval({model.arguments[0]:data_x})

    Save(resave_model_dir, model, data_x, data_y)

    reloaded_model = C.Function.load(resave_model_path, format=C.ModelFormat.ONNX)
    data_y_ = reloaded_model.eval({reloaded_model.arguments[0]:data_x})

    np.testing.assert_equal(len(data_y), len(data_y_))
    for i in range(len(data_y)):
        np.testing.assert_equal(data_y[i].dtype, data_y_[i].dtype)
        np.testing.assert_allclose(
            data_y[i],
            data_y_[i],
            rtol=1e-3,
            atol=1e-4)

    verify_model(model_name, str(os.path.abspath(tmpdir)))