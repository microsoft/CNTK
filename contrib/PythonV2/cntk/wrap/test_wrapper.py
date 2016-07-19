import pytest

import numpy as np
from cntk import cntk_py
from cntk.ops import input, constant

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

    left_var = input(shape, data_type='float', needs_gradient=True, name="left_node")
    right_const = constant(np.ones(shape=shape), data_type='float', name="right_node")

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
