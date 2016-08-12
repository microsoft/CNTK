import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import cntk.cntk_py as cntk_py

# class MyVariable(cntk_py.Variable):
def create_variable(shape, data_type='float', is_sparse=False, needs_gradient=True, name=""):
    if data_type == 'float':
        cntk_type = cntk_py.DataType_Float
    elif data_type == 'double':
        cntk_type = cntk_py.DataType_Double
    else:
        raise ValueError('Type %s is not supported'%data_type)

    return cntk_py.Variable(shape, is_sparse, cntk_type, needs_gradient, name)

def create_ValuePtr(shape, data_type, dev):
    view = cntk_py.NDArrayView(data_type, cntk_py.StorageFormat_Dense, shape, dev)
    return cntk_py.Value(view)

def create_ValuePtr_with_value(shape, data_type, value, dev):
    if data_type == cntk_py.DataType_Float:
        view = cntk_py.NDArrayViewFloat(value, shape, dev)
    elif data_type == cntk_py.DataType_Double:
        view = cntk_py.NDArrayViewDouble(value, shape, dev)
    return cntk_py.Value(view)

def create_ValuePtr_from_NumPy(nd, dev):
    ndav = cntk_py.NDArrayView(nd, dev, False)
    return cntk_py.Value(ndav)

def forward_backward():
    dev = cntk_py.DeviceDescriptor.CPUDevice()

    # import time;time.sleep(20)
    left_shape = (2,3)
    right_shape = (2,3)
    output_shape = (2,3)

    # import time;time.sleep(20)
    left_var = create_variable(left_shape, data_type='float', needs_gradient=True, name="left_node")
    right_var = create_variable(right_shape, data_type='float', needs_gradient=True, name="right_node")

    nd = np.arange(2*3).reshape(left_shape+(1,1)).astype(np.float32)
    left_value_ptr = create_ValuePtr_from_NumPy(nd, dev)
    right_value_ptr = create_ValuePtr_with_value(right_shape+(1,1),
            cntk_py.DataType_Float, 5, dev)

    op = cntk_py.Plus(left_var, right_var)

    outputVariable = op.output()
    output_value_ptr = create_ValuePtr(output_shape+(1,1), cntk_py.DataType_Float, dev) 

    arguments = {} 
    arguments[left_var] = left_value_ptr
    arguments[right_var] = right_value_ptr

    outputs = {} 
    outputs[outputVariable] = output_value_ptr

    outputs_retain = {outputVariable} # cntk_py.VarSet([outputVariable])
    #
    # Forward
    #
    backpropstate = op.forward(arguments, outputs, dev, outputs_retain)
    forward_data = output_value_ptr.data().to_numpy()#.reshape(output_shape)
    print("Result forward:")
    print(forward_data)

    #
    # Backward
    #
    grad_left_value_ptr = create_ValuePtr(left_shape+(1,1),
            cntk_py.DataType_Float, dev)
    grad_right_value_ptr = create_ValuePtr(right_shape+(1,1),
            cntk_py.DataType_Float, dev) 
    gradients = {} 
    gradients[left_var] = grad_left_value_ptr
    gradients[right_var] = grad_right_value_ptr

    rootGradients = {} 
    rootGradientValuePtr = create_ValuePtr_with_value(outputVariable.shape().dimensions()+(1,1), cntk_py.DataType_Float, 1, dev) 
    rootGradients[outputVariable] = rootGradientValuePtr
 
    op.backward(backpropstate, rootGradients, gradients)
    left_grad_data = grad_left_value_ptr.data().to_numpy().reshape(output_shape)
    print("Result backward left:")
    print(left_grad_data)

    right_grad_data = grad_right_value_ptr.data().to_numpy().reshape(output_shape)
    print("Result backward right:")
    print(right_grad_data)

if __name__=='__main__':
    forward_backward()

