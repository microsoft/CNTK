import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import cntk.cntk_py as cntk_py
from cntk.ops import variable, constant, parameter, cross_entropy_with_softmax, combine, classification_error, sigmoid, plus, times

def fully_connected_layer(input, output_dim, device_id, nonlinearity):    
    #TODO: wrap in SWIG or python Shape() methods in order to return the reversed shape (row/col major)
    input_dim = input.Shape()[0]    
    times_param = parameter(shape=(input_dim,output_dim))    
    t = times(input,times_param)
    plus_param = parameter(shape=(output_dim,))
    p = plus(plus_param,t.Output())    
    return nonlinearity(p.Output());

def fully_connected_classifier_net(input, num_output_classes, hidden_layer_dim, num_hidden_layers, device, nonlinearity):
    classifier_root = fully_connected_layer(input, hidden_layer_dim, device, nonlinearity)
    for i in range(1, num_hidden_layers):
        classifier_root = fully_connected_layer(classifier_root.Output(), hidden_layer_dim, device, nonlinearity)
    
    output_times_param = parameter(shape=(hidden_layer_dim,num_output_classes))
    output_plus_param = parameter(shape=(num_output_classes,))
    t = times(classifier_root.Output(),output_times_param)
    classifier_root = plus(output_plus_param,t.Output()) 
    return classifier_root;

def create_minibatch_source():
    #todo: add helper functions
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
    deserializerConfiguration["file"] = cntk_py.DictionaryValue(r"../../../../../Examples/Other/Simple2d/Data/SimpleDataTrain_cntk_text.txt")   
    deserializerConfiguration["input"] = cntk_py.DictionaryValueFromDict(inputStreamsConfig)

    minibatchSourceConfiguration = cntk_py.Dictionary()
    minibatchSourceConfiguration["epochSize"] = cntk_py.DictionaryValue(sys.maxsize)
    deser = cntk_py.DictionaryValueFromDict(deserializerConfiguration)
    minibatchSourceConfiguration["deserializers"] = cntk_py.DictionaryValue([deser])

    return cntk_py.CreateCompositeMinibatchSource(minibatchSourceConfiguration)

if __name__=='__main__':      
    import time;time.sleep(1)   
    input_dim = 2
    num_output_classes = 2
    num_hidden_layers = 2
    hidden_layers_dim = 50
    
    minibatch_size = 25
    num_samples_per_sweep = 10000
    num_sweeps_to_train_with = 2
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size
    lr = 0.02
    input = variable((input_dim,), np.float32, needs_gradient=False, name="features")
    label = variable((num_output_classes,), np.float32, needs_gradient=False, name="labels")
    dev = cntk_py.DeviceDescriptor.CPUDevice()     
    netout = fully_connected_classifier_net(input, num_output_classes, hidden_layers_dim, num_hidden_layers, dev, sigmoid)  
        
    ce = cross_entropy_with_softmax(netout.Output(), label)

    pe = classification_error(netout.Output(), label)
    ffnet = combine([ce, pe, netout], "classifier_model")      
    
    cm = create_minibatch_source()
        
    streamInfos = cm.StreamInfos();    
    
    minibatchSizeLimits = dict()    
    minibatchSizeLimits[streamInfos[0]] = (0,minibatch_size)
    minibatchSizeLimits[streamInfos[1]] = (0,minibatch_size)
                          
    trainer = cntk_py.Trainer(ffnet, ce.Output(), [cntk_py.SGDLearner(ffnet.Parameters(), lr)])          
    
    for i in range(0,int(num_minibatches_to_train)):        
        mb=cm.GetNextMinibatch(minibatchSizeLimits, dev)

        
        arguments = dict()
        arguments[input] = mb[streamInfos[0]].m_data
        arguments[label] = mb[streamInfos[1]].m_data
            
        trainer.TrainMinibatch(arguments, dev)
        freq = 20
        if i % freq == 0:            
            print(str(i+freq) + ": " + str(trainer.PreviousMinibatchTrainingLossValue().Data().ToNumPy()))
    