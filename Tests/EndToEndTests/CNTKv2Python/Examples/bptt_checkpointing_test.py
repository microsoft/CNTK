# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import sys
import signal
import subprocess
import re
import pytest


abs_path = os.path.dirname(os.path.abspath(__file__))
example_dir = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Speech", "AN4", "Python")
sys.path.append(abs_path)
sys.path.append(example_dir)

from HTK_LSTM_Truncated_Distributed import htk_lstm_truncated
from prepare_test_data import an4_dataset_directory


def test_checkpointing_with_truncated_sequences(tmpdir):
  model_path = os.path.join(tmpdir.strpath, 'Models')
  data_path = an4_dataset_directory()
  os.chdir(data_path)
  features_file = os.path.join(data_path, 'glob_0000.scp')
  labels_file = os.path.join(data_path, 'glob_0000.mlf')
  label_mapping_file = os.path.join(data_path, 'state.list')

  htk_lstm_truncated(features_file, labels_file, label_mapping_file,
                    minibatch_size=640, epoch_size=1000, max_epochs=2,
                    model_path=model_path)

  htk_lstm_truncated(features_file, labels_file, label_mapping_file,
                    minibatch_size=640,epoch_size=1000, max_epochs=4,
                    restore=True, model_path=model_path)