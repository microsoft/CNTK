# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import numpy as np
import pytest
import cntk as C

def delete_if_file_exists(file):
    try:
        os.remove(file)
    except OSError:
        pass

def test_saving_int8_ndarray(tmpdir):
    protobuf_file = str(tmpdir/'dictionary_val.bin')
    delete_if_file_exists(protobuf_file)

    data = np.arange(0,64, dtype=np.int8).reshape(16,4)
    dict_val = C._to_cntk_dict_value(data)
    dict_val.save(protobuf_file)

    assert(os.path.getsize(protobuf_file) == 82)

    a = dict_val.load(protobuf_file)
    assert(a==dict_val)

def test_saving_and_loading_int8_ndarray_as_attribute(tmpdir):
    model_file = str(tmpdir/'test_model.bin')
    delete_if_file_exists(model_file)

    data = np.arange(0,64, dtype=np.int8).reshape(16,4)
    dict_val = C._to_cntk_dict_value(data)

    W = C.Parameter((C.InferredDimension, 42), init=C.glorot_uniform(), dtype=np.float)
    x = C.input_variable(12, dtype=np.float)
    y = C.times(x, W)
    y.custom_attributes = {'int8_nd':dict_val}
    y.save(model_file)

    assert(os.path.isfile(model_file))

    z = C.load_model(model_file)
    int8_data = z.custom_attributes['int8_nd']
    assert(int8_data.shape == (16,4))

    assert (np.array_equal(int8_data, data))
