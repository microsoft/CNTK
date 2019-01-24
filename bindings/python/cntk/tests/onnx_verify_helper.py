# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os, re, sys, subprocess

windows = os.getenv("OS")=="Windows_NT"

known_issues = [
    'BatchNormalization_float160',
    'SpatialBatchNormalization_float160',
    'RNN.reverse.one_layer.relu',
    'RNN.bidirectional.two_layer.tanh',
    'SequenceSoftmax',

    # Not in onnxruntime
    'LayerNorm_0',
    'MVN_0',
    'MVN_1',
    'MVN_2',
    'MVN_3',
    'Eye_Like_0',

    # ConstantOfShape not in onnxruntime
    'SequenceIsFirst',
    'SequenceIsLast',
    'Zeros_Like_0',
    'Ones_Like_0',

    # OneHot not in onnxruntime
    'One_Hot_0',
    'One_Hot_1',
]

def parse_single_result_case(case_str):
    fails = re.search(r'Failed Test Cases:[\w\.\-]+', case_str)
    if fails:
        failed_case = fails.group().split(':')[1]
        if not failed_case in known_issues:
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

def verify_model(model_name, model_dir):
    path_prefix = os.path.join(os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'], 'ONNXRuntime') if 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY' in os.environ else ''
    onnx_test_runner_path_str = str(os.path.join(path_prefix, 'onnx_test_runner.exe'))
    # run only on windows. 
    if not os.path.exists(onnx_test_runner_path_str) or not windows:
        return 0
    callargs = [onnx_test_runner_path_str, '-n', model_name, str(model_dir)]
    process = subprocess.run(callargs, stdout=subprocess.PIPE)
    return parse_verify_out_str(process.stdout.decode('utf-8'))

def get_onnx_test_runner_callscript(model_name, model_dir):
    return R'onnx_test_runner.exe -n ' + model_name + ' ' + str(model_dir)