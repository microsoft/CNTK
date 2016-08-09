import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import cntk.cntk_py as cntk_py

#TODO: Make use of the helper functions and move to row major.

def create_variable(shape, data_type='float', is_sparse=False, needs_gradient=True, name=""):
    if data_type == 'float':
        cntk_type = cntk_py.DataType_Float
    elif data_type == 'double':
        cntk_type = cntk_py.DataType_Double
    else:
        raise ValueError('Type %s is not supported'%data_type)

    return cntk_py.Variable(shape, is_sparse, cntk_type, needs_gradient, name)

def fully_connected_layer(input, output_dim, device_id, nonlinearity):    
    input_dim = input.Shape()[0]    
    #import ipdb;ipdb.set_trace()        
    v1 = cntk_py.NDArrayView.RandomUniformFloat((output_dim,input_dim), -0.05, 0.05, 1, device_id)    
    times_param = cntk_py.Parameter(v1)    
    t = cntk_py.Times(times_param, input)    
    v2 = cntk_py.NDArrayView.RandomUniformFloat((output_dim,), -0.05, 0.05, 1, device_id)       
    plus_param = cntk_py.Parameter(v2)
    p = cntk_py.Plus(plus_param,t.Output())    
    return nonlinearity(p.Output());

def fully_connected_classifier_net(input, num_output_classes, hidden_layer_dim, num_hidden_layers, device, nonlinearity):
    scaling_factor = cntk_py.ConstantFloat((), 0.00390625, device)
    scaled_input = cntk_py.ElementTimes(scaling_factor, input)
    classifier_root = fully_connected_layer(scaled_input.Output(), hidden_layer_dim, device, nonlinearity)
    for i in range(1, num_hidden_layers):
        classifier_root = fully_connected_layer(classifier_root.Output(), hidden_layer_dim, device, nonlinearity)
    
    v1 = cntk_py.NDArrayView.RandomUniformFloat((num_output_classes,hidden_layer_dim), -0.05, 0.05, 1, device)    
    output_times_param = cntk_py.Parameter(v1)    

    v2 = cntk_py.NDArrayView.RandomUniformFloat((num_output_classes,), -0.05, 0.05, 1, device)       
    output_plus_param = cntk_py.Parameter(v2)
    t = cntk_py.Times(output_times_param, classifier_root.Output())
    classifier_root = cntk_py.Plus(output_plus_param,t.Output()) 
    return classifier_root;

if __name__=='__main__':      
    import time;time.sleep(1)
    dev = cntk_py.DeviceDescriptor.CPUDevice()       
    input_dim = 784
    num_output_classes = 10
    num_hidden_layers = 1
    hidden_layers_dim = 200
    
    minibatch_size = 32
    num_samples_per_sweep = 60000
    num_sweeps_to_train_with = 3
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size
    lr = 0.003125
    input = create_variable((input_dim,), needs_gradient=False, name="features")
    label = create_variable((num_output_classes,), needs_gradient=False, name="labels")
    
    netout = fully_connected_classifier_net(input, num_output_classes, hidden_layers_dim, num_hidden_layers, dev, cntk_py.Sigmoid)  
        
    ce = cntk_py.CrossEntropyWithSoftmax(netout.Output(), label)
    pe = cntk_py.ClassificationError(netout.Output(), label)
    ffnet = cntk_py.Combine([ce, pe, netout], "classifier_model")      
    
    featuresStreamConfig = cntk_py.Dictionary();
    featuresStreamConfig["dim"] = cntk_py.DictionaryValue(input_dim)       
    featuresStreamConfig["format"] = cntk_py.DictionaryValue("dense")

    labelsStreamConfig = cntk_py.Dictionary()
    labelsStreamConfig["dim"] = cntk_py.DictionaryValue(num_output_classes)
    labelsStreamConfig["format"] = cntk_py.DictionaryValue("dense")

    inputStreamsConfig = cntk_py.Dictionary()
    inputStreamsConfig["features"] = cntk_py.DictionaryValueFromDict(featuresStreamConfig)
    inputStreamsConfig["labels"] = cntk_py.DictionaryValueFromDict(labelsStreamConfig)

    deserializerConfiguration = cntk_py.Dictionary()
    deserializerConfiguration["type"] = cntk_py.DictionaryValue("CNTKTextFormatDeserializer")
    deserializerConfiguration["module"] = cntk_py.DictionaryValue("CNTKTextFormatReader")    
    deserializerConfiguration["file"] = cntk_py.DictionaryValue(r"E:\CNTK\contrib\Python\cntk\examples\MNIST\Data\Train-28x28_text.txt")
    deserializerConfiguration["input"] = cntk_py.DictionaryValueFromDict(inputStreamsConfig)

    minibatchSourceConfiguration = cntk_py.Dictionary()
    minibatchSourceConfiguration["epochSize"] = cntk_py.DictionaryValue(num_samples_per_sweep)
    deser = cntk_py.DictionaryValueFromDict(deserializerConfiguration)
    minibatchSourceConfiguration["deserializers"] = cntk_py.DictionaryValue([deser])

    cm = cntk_py.CreateCompositeMinibatchSource(minibatchSourceConfiguration)
        
    streamInfos = cm.StreamInfos();    
    
    minibatchSizeLimits = dict()    
    minibatchSizeLimits[streamInfos[0]] = (0,minibatch_size)
    minibatchSizeLimits[streamInfos[1]] = (0,minibatch_size)
                
    mb=cm.GetNextMinibatch(minibatchSizeLimits, dev)
      
    trainer = cntk_py.Trainer(ffnet, ce.Output(), [cntk_py.SGDLearner(ffnet.Parameters(), lr)])          
    
    for i in range(0,int(num_minibatches_to_train)):    
        a=cm.GetNextMinibatch(minibatchSizeLimits, dev)

        
        arguments = dict()
        arguments[input] = mb[streamInfos[0]].m_data
        arguments[label] = mb[streamInfos[1]].m_data

        
        trainer.TrainMinibatch(arguments, dev)
        freq = 20
        if i % freq == 0:
            print(str(i+freq) + ": " + str(trainer.PreviousMinibatchTrainingLossValue().Data().ToNumPy()))
    