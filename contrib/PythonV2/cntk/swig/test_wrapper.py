import pytest
import sys
import numpy as np
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import cntk.cntk_py as cntk_py
from cntk.ops import variable, constant
from cntk.tests.test_utils import *
from cntk.utils import sanitize_batch, remove_masked_elements, pad_to_dense, precision_numpy, cntk_device

def create_Value_with_value(shape, data_type, value, dev):
    if data_type == cntk_py.DataType_Float:
        view = cntk_py.NDArrayViewFloat(value, shape, dev)
    elif data_type == cntk_py.DataType_Double:
        view = cntk_py.NDArrayViewDouble(value, shape, dev)
    return cntk_py.Value(view)

def create_Value_from_NumPy(nd, dev):
    ndav = cntk_py.NDArrayView(nd, dev, False)
    return cntk_py.Value(ndav)

def _input_plus_const():
    dev = cntk_py.DeviceDescriptor.CPUDevice()

    shape = (2,3)

    left_var = variable(shape, data_type='float', needs_gradient=True, name="left_node")
    right_const = constant(value=np.ones(shape=shape), name="right_node")

    op = cntk_py.Plus(left_var, right_const)

    return left_var, right_const, op

def test_hashability_1():
    left, right, op = _input_plus_const()
    assert left != right

    outputVariables = op.Outputs()
    assert len(outputVariables) == 1
    assert op.Output() in op.Outputs()

    # Swig creates always new references when underlying objects are requested.
    # Here, we check whether the hashing is done properly on the C++ side.
    for o in outputVariables:
        assert o in op.Outputs()

    # constants are counted as inputs
    assert len(op.Inputs()) == 2
    assert left in op.Inputs()
    assert right in op.Inputs()

    assert len(op.Constants()) == 1
    assert left not in op.Constants()
    assert right in op.Constants()

    assert len(op.Placeholders()) == 0

    assert len(op.Parameters()) == 0

def test_masking(device_id, precision):    
    # Batch of three sequences of lengths 2, 4, and 1
    batch = [
            [[1], [2]],
            [[3], [4], [5], [6]],
            [[7]]
            ]

    for idx,sample in enumerate(batch):
        batch[idx] = AA(sample)

    num_samples = len(batch)
    max_seq_len = max(len(seq) for seq in batch)
            
    precision_np = precision_numpy(precision)    
    device = cntk_device(device_id)
            
    # sanitizing the batch will wrap it into a Value with a corresponding mask
    value = sanitize_batch(batch, precision_np, device)

    mask = value.Mask()

    # mask_array will contain a row per sequence, with the columns having 1 for valid
    # entries and 0 for invalid ones.
    mask_array = mask.ToNumPy()
    assert mask_array.shape == (num_samples, max_seq_len)

    assert np.all(mask_array[0] == AA([1, 1, 0, 0]))
    assert np.all(mask_array[1] == AA([1, 1, 1, 1]))
    assert np.all(mask_array[2] == AA([1, 0, 0, 0]))

    # now pad the batch and check whether our helper function correctly uses
    # the mask in order to remove the padded zeros
    list_of_np = remove_masked_elements(pad_to_dense(batch), mask_array)
    for actual, expected in zip(list_of_np, batch):
        assert np.all(actual==expected)


