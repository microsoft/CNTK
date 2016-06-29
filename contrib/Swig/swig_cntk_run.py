import sys
sys.path.append(r"C:\blis\CNTK\x64\Release_CpuOnly")
import swig_cntk

def shape_from_NDShape(swig_obj):
    shape =  []
    for i in range(swig_obj.NumAxes()):
        shape.append(swig_obj[i])

    return tuple(shape)

def create_variable(shape, data_type=float, is_sparse=False, needs_gradient=True, name=""):
    if data_type == float:
        cntk_type = swig_cntk.DataType_Float
    elif data_type == double:
        cntk_type = swig_cntk.DataType_Double
    else:
        raise ValueError('Type %s is not supported'%data_type)

    return swig_cntk.Variable(swig_cntk.NDShape(shape), cntk_type, is_sparse,  name)

def create_ValuePtr(shape, data_type, dev):
    ndshape = swig_cntk.NDShape(shape+(1,1))
    view = swig_cntk.NDArrayViewFloat(data_type, ndshape, dev)
    view_ptr = swig_cntk.NDArrayViewPtr(view)
    value = swig_cntk.Value(view_ptr)
    return swig_cntk.ValuePtr(value)

def create_ValuePtr_with_value(shape, value, dev):
    ndshape = swig_cntk.NDShape(shape+(1,1))
    view = swig_cntk.NDArrayViewFloat(value, ndshape, dev)
    view_ptr = swig_cntk.NDArrayViewPtr(view)
    value = swig_cntk.Value(view_ptr)
    return swig_cntk.ValuePtr(value)

def cntk():
    dev = swig_cntk.DeviceDescriptor.CPUDevice()

    left_shape = (2,3)
    right_shape = (2,3)
    output_shape = (2,3)

    # import time;time.sleep(20)
    left_var = create_variable(left_shape, name="left_node")
    right_var = create_variable(right_shape, name="right_node")

    left_value_ptr = create_ValuePtr_with_value(left_shape, 2, dev)
    right_value_ptr = create_ValuePtr_with_value(right_shape, 5, dev)

    op = swig_cntk.Plus(left_var, right_var)

    outputVariable = op.Output()
    output_shape = (2,3)
    output_value_ptr = create_ValuePtr(output_shape, swig_cntk.DataType_Float, dev) 

    arguments = swig_cntk.MapVarValuePtr()
    arguments[left_var] = left_value_ptr
    arguments[right_var] = right_value_ptr

    outputs = swig_cntk.MapVarValuePtr()
    outputs[outputVariable] = output_value_ptr

    outputs_to_retain = set()
    backpropstate = op.ForwardMap(arguments, outputs, dev)#, outputs_to_retain)

    import numpy as np
    output_ndshape = output_value_ptr.Data().Shape() 
    data = swig_cntk.data_from_value(
            output_value_ptr.Data().GetPtr().DataBufferFloat(), 
            output_ndshape.TotalSize())

    return data.reshape(output_shape)

if __name__=='__main__':
    res = cntk()
    print(res)
