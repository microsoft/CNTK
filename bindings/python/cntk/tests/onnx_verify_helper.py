# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os, re, sys, subprocess, scipy, pytest, numpy as np
import cntk as C
onnx = pytest.importorskip("onnx")

windows = os.getenv("OS")=="Windows_NT"

known_issues = [
    'SpatialBatchNormalization_float160',
    'RNN.reverse.one_layer.relu',

    # onnxruntime supports only [NCHW] for mvn.
    'LayerNorm_0(10_)',
    'LayerNorm_0(20_ 31)',
]

def parse_single_result_case(case_str):
    fails = re.search(r'Failed Test Cases:[\w\.\-\_\(\)\s]+', case_str)
    if fails:
        failed_case = fails.group().split(':')[1]
        if not failed_case in known_issues and not failed_case:
            print(case_str, file=sys.stderr)
            return 1
    return 0

def parse_verify_out_str(content):
    total_failed_cases = 0

    case_list = re.findall(r'result:[\s\S]*?Failed Test Cases:[^\n]*\n', content)
    for case_str in case_list:
        total_failed_cases += parse_single_result_case(case_str)

    if total_failed_cases:
        print('ERROR: onnx_test_runner produced ' + str(total_failed_cases) + ' failed cases.', file=sys.stderr)
        sys.exit(1)

    return total_failed_cases

def get_onnx_test_runner_path_str():
    # find onnx_test_runner.exe from repo source in default. 
    # in case not found (in the build environment), look for based on environment variables. 
    onnx_test_runner_path_str = os.path.join(os.path.dirname(__file__), 'onnx_test_runner') + R'/onnx_test_runner.exe';
    if not os.path.exists(onnx_test_runner_path_str):
        workspace_path = None
        if 'BUILD_REPOSITORY_LOCALPATH' in os.environ:
            workspace_path = os.environ['BUILD_REPOSITORY_LOCALPATH']
        elif 'WORKSPACE' in os.environ:
            workspace_path = os.environ['WORKSPACE']
        elif 'WINDOWS_WORKSPACE' in os.environ:
            workspace_path = os.environ['WINDOWS_WORKSPACE']

        assert (workspace_path, 'None of BUILD_REPOSITORY_LOCALPATH, WORKSPACE, or WINDOWS_WORKSPACE is set in os.environ')
        onnx_test_runner_path_str = workspace_path + R'/bindings/python/cntk/tests/onnx_test_runner/onnx_test_runner.exe';

    assert(os.path.exists(onnx_test_runner_path_str), onnx_test_runner_path_str + ' does not exist.')
    return onnx_test_runner_path_str

def verify_results_with_onnxruntime(model_name, model_dir):
    if not windows:
        return 0
    onnx_test_runner_path_str = get_onnx_test_runner_path_str()
    callargs = [onnx_test_runner_path_str, '-n', model_name, str(model_dir)]
    process = subprocess.run(callargs, stdout=subprocess.PIPE)
    return parse_verify_out_str(process.stdout.decode('utf-8'))

def get_onnx_test_runner_callscript(model_name, model_dir):
    return R'onnx_test_runner.exe -n ' + model_name + ' ' + str(model_dir)

def generate_sequence_data(batch_size, seq_len, feature_size, input_as_index = False):
    assert batch_size == 1
    np.random.seed(0)
    data = np.zeros((batch_size, seq_len)).astype(np.float32) if input_as_index else np.zeros((batch_size, seq_len, feature_size)).astype(np.float32) 
    for i in range(0,seq_len):
        one_hot_index = np.random.random_integers(0, feature_size - 1)
        if input_as_index:
            data[0][i] = one_hot_index
        else:
            data[0][i][one_hot_index] = 1
    return data

def generate_sequential_data(tensor_shape):
    total = np.prod(tensor_shape)
    return np.reshape(range(0, total), tensor_shape).astype(np.float32)

def generate_sparse_data(batch_size, seq_len, feature_size):
    sparse_data = []
    for batch in range(0, batch_size):
        data = np.zeros((seq_len, feature_size)).astype(np.float32)
        np.random.seed(0)
        for i in range(0,seq_len): 
            one_hot_index = np.random.random_integers(0, feature_size - 1)
            data[i][one_hot_index] = 1.0
        sparse_data.append(scipy.sparse.csr_matrix(data))
    return sparse_data

def generate_sparse_data_non_seq(batch_size, feature_size):
    data = np.zeros((batch_size, feature_size)).astype(np.float32)
    np.random.seed(0)
    for i in range(0,batch_size): 
        one_hot_index = np.random.random_integers(0, feature_size - 1)
        data[i][one_hot_index] = 1.0
    sparse_data = scipy.sparse.csr_matrix(data)
    return sparse_data