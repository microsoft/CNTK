import numpy as np
import sys
import os
import cntk_py

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

def create_ValuePtr_from_NumPy2(ndav):    
    return cntk_py.Value(ndav)

def fully_connected_layer(input, output_dim, device_id, nonlinearity):    
    input_dim = input.Shape()[0]    
    #import ipdb;ipdb.set_trace()        
    v = cntk_py.NDArrayView.RandomUniformFloat((output_dim,input_dim), -0.5, 0.5, 1, device_id)    
    times_param = cntk_py.Parameter(v)    
    t = cntk_py.Times(times_param, input)    
    view = cntk_py.NDArrayViewFloat(0, (output_dim,), dev)            
    plus_param = cntk_py.Parameter(view)
    p = cntk_py.Plus(t.Output(), plus_param)    
    return nonlinearity(p.Output());

def fully_connected_classifier_net(input, num_output_classes, hidden_layer_dim, num_hidden_layers, device, nonlinearity):
    classifier_root = fully_connected_layer(input, hidden_layer_dim, device, nonlinearity);
    for i in range(1, num_hidden_layers):
        classifier_root = fully_connected_layer(classifier_root.Output(), hidden_layer_dim, device, nonlinearity);

    #view_ptr = cntk_py.NDArrayViewPtr()
    v = cntk_py.NDArrayView.RandomUniformFloat((num_output_classes,hidden_layer_dim), -0.5, 0.5, 1, device)    
    output_times_param = cntk_py.Parameter(v)    
    classifier_root = cntk_py.Times(output_times_param, classifier_root.Output());
    return classifier_root;


if __name__=='__main__':      
    dev = cntk_py.DeviceDescriptor.CPUDevice()       
    input_dim = 937;
    num_output_classes = 9304;
    num_hidden_layers = 12;
    hidden_layers_dim = 2048;
    num_samples = 5000;
    
    
    
    input = create_variable((input_dim,), needs_gradient=True, name="input")
    label = create_variable((num_output_classes,), needs_gradient=True, name="label")
    
    netout = fully_connected_classifier_net(input, num_output_classes, hidden_layers_dim, num_hidden_layers, dev, cntk_py.Sigmoid)  
        
    ce = cntk_py.CrossEntropyWithSoftmax(netout.Output(), label)
    pe = cntk_py.ClassificationError(netout.Output(), label)
    ffnet = cntk_py.Combine([ce, pe, netout], "aa")
    
    for i in range(0,1):
        nd = np.random.rand(input_dim,1,num_samples)        
        input_value_ptr = create_ValuePtr_from_NumPy(nd.astype(np.float32), dev)
              
        label_data = np.zeros(num_output_classes*num_samples)
        for j in range(0, num_samples):
            label_data[(j*num_output_classes)+np.random.randint(0,num_output_classes)] = 1
        label_data = label_data.reshape((num_output_classes,)+(1,num_samples)).astype(np.float32)
        label_value_ptr = create_ValuePtr_from_NumPy(label_data, dev)

        arguments = dict()
        arguments[input] = input_value_ptr
        arguments[label] = label_value_ptr

        netout_variable = netout.Output()
        prediction_err_var = pe.Output()
        output_variable = ce.Output()

        
        output_shape = (1,)
        output_value_ptr = create_ValuePtr((), cntk_py.DataType_Float, dev) 
        prediction_err__value_ptr = create_ValuePtr((), cntk_py.DataType_Float, dev) 
        netout_value_ptr = create_ValuePtr((num_output_classes,)+(1,num_samples), cntk_py.DataType_Float, dev) 

        outputs = dict()
        outputs[netout_variable] = netout_value_ptr
        outputs[prediction_err_var] = prediction_err__value_ptr        
        outputs_retain = set([output_variable])     

        #
        # Forward
        #
        
        backpropstate = ffnet.Forward(arguments, outputs, dev, outputs_retain)
                       
        forward_data = output_value_ptr.Data().ToNumPy().reshape(output_shape)

        grad_input_value_ptr = create_ValuePtr((input_dim,)+(1,num_samples), cntk_py.DataType_Float, dev)
        
        gradients = dict()
        gradients[input] = grad_input_value_ptr
        
        rootGradients = dict()

        rootGradientValuePtr = create_ValuePtr_with_value((), cntk_py.DataType_Float, 1, dev) 
        rootGradients[output_variable] = rootGradientValuePtr
     
        ffnet.Backward(backpropstate, rootGradients, gradients)
        input_grad_data = grad_input_value_ptr.Data().ToNumPy()
        

        

