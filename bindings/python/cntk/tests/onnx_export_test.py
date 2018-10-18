# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os, re, sys

onnxruntime_verify_out_path = 'tmp_exported_models/onnxruntime_verify_out.txt'

known_issues = [
    # 'BatchNormalization_float160',
    # 'cast_float16_to_float16',
    # 'ConvTranspose_with_OutputShape_0',
    # 'DepthToSpace',
    # 'Flatten_1',
    # 'Gather_1',
    # 'LayerNorm_0',
    # 'MVN_0',
    # 'MVN_1',
    # 'MVN_2',
    # 'MVN_3',
    # 'RNN',
    # 'test_sequence_slice_',
    # 'test_sequence_slice_0',
    # 'test_sequence_slice_1',
    # 'SequenceSoftmax',
    # 'SpaceToDepth',
    # 'top_k',
]

def parse_single_result_case(case_str):
    fails = re.search(r'Failed Test Cases:\w+', case_str)
    if fails:
        failed_case = fails.group().split(':')[1]
        if not failed_case in known_issues:
            print(case_str, file=sys.stderr)
            return 1
    return 0

def parse_verify_out_file(filepath):
    total_failed_cases = 0
    with open(filepath, 'r', encoding='utf-16') as file:
        content = file.read()

        case_list = re.findall(r'result:[\s\S]*?Failed Test Cases:[^\n]*\n', content)
        
        for case_str in case_list:
            total_failed_cases += parse_single_result_case(case_str)

        if total_failed_cases:
            print('ERROR: onnx_test_runner produced ' + str(total_failed_cases) + ' failed cases.', file=sys.stderr)
            sys.exit(1)

        return total_failed_cases

parse_verify_out_file(onnxruntime_verify_out_path)