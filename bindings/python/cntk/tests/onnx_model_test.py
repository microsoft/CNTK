# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import cntk as C
import numpy as np
import pytest
import os
import re

onnx = pytest.importorskip("onnx")
from onnx import numpy_helper

# To test models locally, create folder 'onnx_models' and put in model folders. 
# For example.
#   .
#   +-- onnx_models
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
base_dir = 'onnx_models' if not 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY' in os.environ else os.path.join(os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'], 'onnx_models')
model_names = [dir for dir in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, dir))] if os.path.exists(base_dir) else []
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
    'test_max_example',
    'test_min_example',
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
    'test_averagepool_2d_pads',
    'test_maxpool_2d_pads',
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
    'test_max_example',
    'test_min_example',
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
    'test_averagepool_2d_pads',
    'test_maxpool_2d_pads',
]

@pytest.mark.parametrize('model_name, round_trip', 
    [(model_name, round_trip) for model_name in model_names for round_trip in [False, True]], 
    ids=['round_trip_' + model_name if round_trip else model_name for model_name in model_names for round_trip in [False, True]])
def test_onnx_model(model_name, round_trip):
    if model_name in skip_model_names and not round_trip:
        pytest.skip('Skip onnx model test. ')
    if model_name in skip_round_trip_model_names and round_trip:
        pytest.skip('Skip onnx model round trip test. ')

    model_dir = os.path.join(base_dir, model_name)
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