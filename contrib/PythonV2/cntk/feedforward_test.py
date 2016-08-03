import numpy as np
import sys
import os
import cntk_py

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
    v1 = cntk_py.NDArrayView.RandomUniformFloat((output_dim,input_dim), -0.5, 0.5, 1, device_id)    
    times_param = cntk_py.Parameter(v1)    
    t = cntk_py.Times(times_param, input)    
    v2 = cntk_py.NDArrayView.RandomUniformFloat((output_dim,), -0.5, 0.5, 1, device_id)       
    plus_param = cntk_py.Parameter(v2)
    p = cntk_py.Plus(t.Output(), plus_param)    
    return nonlinearity(p.Output());

def fully_connected_classifier_net(input, num_output_classes, hidden_layer_dim, num_hidden_layers, device, nonlinearity):
    classifier_root = fully_connected_layer(input, hidden_layer_dim, device, nonlinearity)
    for i in range(1, num_hidden_layers):
        classifier_root = fully_connected_layer(classifier_root.Output(), hidden_layer_dim, device, nonlinearity)
    
    v1 = cntk_py.NDArrayView.RandomUniformFloat((num_output_classes,hidden_layer_dim), -0.5, 0.5, 1, device)    
    output_times_param = cntk_py.Parameter(v1)    

    v2 = cntk_py.NDArrayView.RandomUniformFloat((num_output_classes,), -0.5, 0.5, 1, device)       
    output_plus_param = cntk_py.Parameter(v2)
    t = cntk_py.Times(output_times_param, classifier_root.Output())
    classifier_root = cntk_py.Plus(t.Output(), output_plus_param) 
    return classifier_root;

if __name__=='__main__':      
    dev = cntk_py.DeviceDescriptor.CPUDevice()       
    input_dim = 2
    num_output_classes = 2
    num_hidden_layers = 2
    hidden_layers_dim = 50
    
    minibatch_size = 25
    num_samples_per_sweep = 10000
    num_sweeps_to_train_with = 2
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size
    
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
    deserializerConfiguration["file"] = cntk_py.DictionaryValue("SimpleDataTrain_cntk_text.txt")
    deserializerConfiguration["input"] = cntk_py.DictionaryValueFromDict(inputStreamsConfig)

    minibatchSourceConfiguration = cntk_py.Dictionary()
    minibatchSourceConfiguration["epochSize"] = cntk_py.DictionaryValue(num_samples_per_sweep)
    deser = cntk_py.DictionaryValueFromDict(deserializerConfiguration)
    minibatchSourceConfiguration["deserializers"] = cntk_py.DictionaryValue([deser])

    cm = cntk_py.CreateCompositeMinibatchSource(minibatchSourceConfiguration)
        
    streamInfos = cm.StreamInfos();    

    minibatchData = dict()    
    minibatchData[streamInfos[0]] = (minibatch_size, None)
    minibatchData[streamInfos[1]] = (minibatch_size, None)
                
    trainer = cntk_py.Trainer(ffnet, ce.Output(), [cntk_py.SGDLearner(ffnet.Parameters(), 0.02)])    
    
    for i in range(0,int(num_minibatches_to_train)):
            
        # TODO: Fix this, for some reason we need to reset minibatch_size in the dictionary becuase
        # it is getting messed up by SWIG resulting in the following CNTK error:
        # "Different minibatch sizes across different input streams is currently unsupported!"
        # it should be an easy fix in SWIG
        minibatchData[streamInfos[0]] = (minibatch_size, minibatchData[streamInfos[0]][1])
        minibatchData[streamInfos[1]] = (minibatch_size, minibatchData[streamInfos[1]][1])        
        cm.GetNextMinibatch(minibatchData)
                
        arguments = dict()
        arguments[input] = minibatchData[streamInfos[0]][1]
        arguments[label] = minibatchData[streamInfos[1]][1]
        
        trainer.TrainMinibatch(arguments, dev)
        if i % 20 == 0:
            print(str(i) + ": " + str(trainer.PreviousMinibatchTrainingLossValue().Data().ToNumPy()))