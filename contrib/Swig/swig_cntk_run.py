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

    return swig_cntk.Variable(swig_cntk.NDShape(shape), is_sparse, cntk_type, needs_gradient, name)

def create_ValuePtr(shape, data_type, dev):
    ndshape = swig_cntk.NDShape(shape)
    view = swig_cntk.NDArrayViewFloat(data_type, ndshape, dev)
    view_ptr = swig_cntk.NDArrayViewPtr(view)
    value = swig_cntk.Value(view_ptr)
    return swig_cntk.ValuePtr(value)

def create_ValuePtr_with_value(shape, value, dev):
    ndshape = swig_cntk.NDShape(shape)
    view = swig_cntk.NDArrayViewFloat(value, ndshape, dev)
    view_ptr = swig_cntk.NDArrayViewPtr(view)
    value = swig_cntk.Value(view_ptr)
    return swig_cntk.ValuePtr(value)

def create_ValuePtr_with_Buffer(shape, data_type, dev):
    ndshape = swig_cntk.NDShape(shape)
    view = swig_cntk.NDArrayViewFloat(value, ndshape, dev)
    view_ptr = swig_cntk.NDArrayViewPtr(view)
    value = swig_cntk.Value(view_ptr)
    return swig_cntk.ValuePtr(value)

def forward_backward():
    dev = swig_cntk.DeviceDescriptor.CPUDevice()

    #import time;time.sleep(20)
    left_shape = (2,3)
    right_shape = (2,3)
    output_shape = (2,3)

    # import time;time.sleep(20)
    left_var = create_variable(left_shape, needs_gradient=True, name="left_node")
    right_var = create_variable(right_shape, needs_gradient=True, name="right_node")

    left_value_ptr = create_ValuePtr_with_value(left_shape+(1,1), 2, dev)
    right_value_ptr = create_ValuePtr_with_value(right_shape+(1,1), 5, dev)

    op = swig_cntk.Plus(left_var, right_var)

    outputVariable = op.Output()
    output_shape = (2,3)
    output_value_ptr = create_ValuePtr(output_shape+(1,1), swig_cntk.DataType_Float, dev) 

    arguments = swig_cntk.MapVarValuePtr()
    arguments[left_var] = left_value_ptr
    arguments[right_var] = right_value_ptr

    outputs = swig_cntk.MapVarValuePtr()
    outputs[outputVariable] = output_value_ptr

    outputs_retain = swig_cntk.VarSet([outputVariable])
    #
    # Forward
    #
    backpropstate = op.ForwardMap(arguments, outputs, dev, outputs_retain)
    forward_data = swig_cntk.data_from_value(
            output_value_ptr.Data().GetPtr().DataBufferFloat(), 
            output_value_ptr.Data().Shape().TotalSize()).reshape(output_shape)
    print("Result forward:")
    print(forward_data)

    #
    # Backward
    #
    grad_left_value_ptr = create_ValuePtr(left_shape+(1,1), swig_cntk.DataType_Float, dev)
    grad_right_value_ptr = create_ValuePtr(right_shape+(1,1), swig_cntk.DataType_Float, dev)

    gradients = swig_cntk.MapVarValuePtr()
    gradients[left_var] = grad_left_value_ptr
    gradients[right_var] = grad_right_value_ptr

    rootGradients = swig_cntk.MapVarValuePtr()
    rootGradientValuePtr = create_ValuePtr_with_value(
            shape_from_NDShape(outputVariable.Shape())+(1,1), 1, dev) 
    rootGradients[outputVariable] = rootGradientValuePtr
 
    op.BackwardMap(backpropstate, rootGradients, gradients)
    left_grad_data = swig_cntk.data_from_value(
            grad_left_value_ptr.Data().GetPtr().DataBufferFloat(), 
            grad_left_value_ptr.Data().Shape().TotalSize()).reshape(output_shape)
    print("Result backward left:")
    print(left_grad_data)

    right_grad_data = swig_cntk.data_from_value(
            grad_right_value_ptr.Data().GetPtr().DataBufferFloat(), 
            grad_right_value_ptr.Data().Shape().TotalSize()).reshape(output_shape)
    print("Result backward right:")
    print(right_grad_data)

if __name__=='__main__':
    forward_backward()
