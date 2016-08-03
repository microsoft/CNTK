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
    print ('============================')
    print(classifier_root)
    return classifier_root;


if __name__=='__main__':      
    dev = cntk_py.DeviceDescriptor.CPUDevice()       
    input_dim = 2;
    num_output_classes = 2;
    num_hidden_layers = 2;
    hidden_layers_dim = 50;
    num_samples = 25;
           
    input = create_variable((input_dim,), needs_gradient=True, name="input")
    label = create_variable((num_output_classes,), needs_gradient=True, name="label")
    
    netout = fully_connected_classifier_net(input, num_output_classes, hidden_layers_dim, num_hidden_layers, dev, cntk_py.Sigmoid)  
        
    ce = cntk_py.CrossEntropyWithSoftmax(netout.Output(), label)
    pe = cntk_py.ClassificationError(netout.Output(), label)
    ffnet = cntk_py.Combine([ce, pe, netout], "aa")      
    
    featuresStreamConfig = cntk_py.Dictionary();
    featuresStreamConfig["dim"] = cntk_py.DictionaryValue(input_dim);       
    featuresStreamConfig["format"] = cntk_py.DictionaryValue("dense");

    labelsStreamConfig = cntk_py.Dictionary()
    labelsStreamConfig["dim"] = cntk_py.DictionaryValue(num_output_classes);
    labelsStreamConfig["format"] = cntk_py.DictionaryValue("dense");

    inputStreamsConfig = cntk_py.Dictionary();
    inputStreamsConfig["features"] = cntk_py.DictionaryValueFromDict(featuresStreamConfig);
    inputStreamsConfig["labels"] = cntk_py.DictionaryValueFromDict(labelsStreamConfig);

    deserializerConfiguration = cntk_py.Dictionary();
    deserializerConfiguration["type"] = cntk_py.DictionaryValue("CNTKTextFormatDeserializer");
    deserializerConfiguration["module"] = cntk_py.DictionaryValue("CNTKTextFormatReader");
    deserializerConfiguration["file"] = cntk_py.DictionaryValue(r"E:\CNTK\contrib\PythonV2\cntk\SimpleDataTest_cntk_text.txt");
    deserializerConfiguration["input"] = cntk_py.DictionaryValueFromDict(inputStreamsConfig);

    minibatchSourceConfiguration = cntk_py.Dictionary();
    minibatchSourceConfiguration["epochSize"] = cntk_py.DictionaryValue(10);
    deser = cntk_py.DictionaryValueFromDict(deserializerConfiguration)
    minibatchSourceConfiguration["deserializers"] = cntk_py.DictionaryValue([deser]);

    cm = cntk_py.CreateCompositeMinibatchSource(minibatchSourceConfiguration);
    print ('---------------------------------')
    print (cm)
    streamInfos = cm.StreamInfos();
    print (streamInfos[0].m_name)
    print (streamInfos[1].m_name)
    minibatchData = dict()
    minibatchData[streamInfos[0]] = (10, None)
    minibatchData[streamInfos[1]] = (10, None)
    cm.GetNextMinibatch(minibatchData)
    trainer = cntk_py.Trainer(ffnet, ce.Output(), [cntk_py.SGDLearner(ffnet.Parameters(), 0.2)])    
    for i in range(0,10):
        nd = np.random.rand(input_dim,1,num_samples)        
        
        input_value_ptr = create_ValuePtr_from_NumPy(nd.astype(np.float32), dev)
              
        label_data = np.zeros(num_output_classes*num_samples)
        label_data = label_data.reshape((num_output_classes,)+(1,num_samples)).astype(np.float32)
        for j in range(0, num_samples):           
            if nd[0][0][j] < 0.5:
                label_data[0][0][j] = 1
            else:
                label_data[1][0][j] = 1
        

        
        label_value_ptr = create_ValuePtr_from_NumPy(label_data, dev)

        arguments = dict()
        arguments[input] = input_value_ptr
        arguments[label] = label_value_ptr
        
        trainer.TrainMinibatch(arguments, dev)
        print(trainer.PreviousMinibatchTrainingLossValue().Data().ToNumPy())
