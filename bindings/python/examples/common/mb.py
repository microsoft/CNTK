# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
from cntk.utils import create_minibatch_source

def create_text_mb_source(data_file, input_dim, num_output_classes, epoch_size,
                          is_feature_sparse=False, is_label_sparse=False,
                          feature_alias=None, label_alias=None):
    features_config = dict()
    features_config["dim"] = input_dim
    features_config["format"] = "sparse" if is_feature_sparse else "dense"
    if feature_alias:
        features_config["alias"] = feature_alias

    labels_config = dict()
    labels_config["dim"] = num_output_classes
    labels_config["format"] = "sparse" if is_label_sparse else "dense"
    if label_alias:
        labels_config["alias"] = label_alias

    input_config = dict()
    input_config["features"] = features_config
    input_config["labels"] = labels_config

    deserializer_config = dict()
    deserializer_config["type"] = "CNTKTextFormatDeserializer"
    deserializer_config["module"] = "CNTKTextFormatReader"
    deserializer_config["file"] = data_file
    deserializer_config["input"] = input_config

    minibatch_config = dict()
    minibatch_config["epochSize"] = epoch_size
    minibatch_config["deserializers"] = [deserializer_config]

    return create_minibatch_source(minibatch_config)