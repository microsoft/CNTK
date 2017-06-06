# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os, sys
import numpy as np
import pytest

from cntk.device import try_set_default_device
from cntk.cntk_py import DeviceKind_GPU
from cntk.ops.tests.ops_test_utils import cntk_device

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
base_path = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Text", "WordLMWithSampledSoftmax")
sys.path.append(base_path)

from prepare_test_data import prepare_WordLMWithSampledSoftmax_ptb_data
import word_rnn as W
from data_reader import get_count_data

TOLERANCE_ABSOLUTE = 1e-1

def test_data_reader(device_id):
    try_set_default_device(cntk_device(device_id))

    prepare_WordLMWithSampledSoftmax_ptb_data()

    expected_count = 2104

    current_path = os.getcwd()
    os.chdir(os.path.join(base_path))

    try:
        actual_count = get_count_data()
        assert actual_count == expected_count
    finally:
        os.chdir(current_path)

def test_ptb_word_rnn(device_id):
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('This test only runs on GPU')
    try_set_default_device(cntk_device(device_id))

    prepare_WordLMWithSampledSoftmax_ptb_data()

    W.num_epochs = 1
    W.softmax_sample_size = 3
    W.num_layers = 1

    current_path = os.getcwd()
    os.chdir(os.path.join(base_path))

    try:
        error = W.train_lm(testing=True)
        expected_error = 7.109
        assert np.allclose(error, expected_error, atol=TOLERANCE_ABSOLUTE)
    finally:
        os.chdir(current_path)

def test_word_rnn(device_id):
    try_set_default_device(cntk_device(device_id))

    # Just run and verify it does not crash
    # Setting global parameters
    W.use_sampled_softmax = True
    W.softmax_sample_size = 3
    W.use_sparse = True
    W.hidden_dim = 20
    W.num_layers = 2
    W.num_epochs = 1
    W.sequence_length = 3
    W.sequences_per_batch = 2
    W.alpha = 0.75
    W.learning_rate = 0.02
    W.momentum_as_time_constant = 5
    W.clipping_threshold_per_sample = 5.0
    W.segment_sepparator = '<eos>'
    W.num_samples_between_progress_report = 2

    # Get path to data files.
    dir =  os.path.dirname( os.path.abspath(W.__file__))
    W.token_to_id_path            = os.path.join(dir, 'test/token2id.txt')
    W.validation_file_path        = os.path.join(dir, 'test/text.txt')
    W.train_file_path             = os.path.join(dir, 'test/text.txt')
    W.token_frequencies_file_path = os.path.join(dir, 'test/freq.txt')

    W.train_lm(testing=True)
