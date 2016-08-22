import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import cntk.cntk_py as cntk_py
from cntk.ops import variable, constant, parameter, cross_entropy_with_softmax, combine, classification_error, sigmoid, plus, times
from cntk.utils import create_minibatch_source, get_train_loss
from cntk.tests.test_utils import TOLERANCE_ABSOLUTE

def fully_connected_layer(input, output_dim, device_id, nonlinearity):        
    input_dim = input.shape()[0]    
    times_param = parameter(shape=(input_dim,output_dim))    
    t = times(input,times_param)
    plus_param = parameter(shape=(output_dim,))
    p = plus(plus_param,t.output())    
    return nonlinearity(p.output());

def fully_connected_classifier_net(input, num_output_classes, hidden_layer_dim, num_hidden_layers, device, nonlinearity):
    classifier_root = fully_connected_layer(input, hidden_layer_dim, device, nonlinearity)
    for i in range(1, num_hidden_layers):
        classifier_root = fully_connected_layer(classifier_root.output(), hidden_layer_dim, device, nonlinearity)
    
    output_times_param = parameter(shape=(hidden_layer_dim,num_output_classes))
    output_plus_param = parameter(shape=(num_output_classes,))
    t = times(classifier_root.output(),output_times_param)
    classifier_root = plus(output_plus_param,t.output()) 
    return classifier_root;

def create_mb_source(input_dim, num_output_classes, epoch_size):    
    features_config = dict()
    features_config["dim"] = input_dim
    features_config["format"] = "dense"

    labels_config = dict()
    labels_config["dim"] = num_output_classes
    labels_config["format"] = "dense"

    input_config = dict()
    input_config["features"] = features_config
    input_config["labels"] = labels_config

    deserializer_config = dict()
    deserializer_config["type"] = "CNTKTextFormatDeserializer"
    deserializer_config["module"] = "CNTKTextFormatReader"
    rel_path = r"../../../../../Examples/Other/Simple2d/Data/SimpleDataTrain_cntk_text.txt"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)
    deserializer_config["file"] = path
    deserializer_config["input"] = input_config

    minibatch_config = dict()
    minibatch_config["epochSize"] = epoch_size
    minibatch_config["deserializers"] = [deserializer_config]

    return create_minibatch_source(minibatch_config)

def test_ffnet():
    input_dim = 2
    num_output_classes = 2
    num_hidden_layers = 2
    hidden_layers_dim = 50
    epoch_size = sys.maxsize
    minibatch_size = 25
    num_samples_per_sweep = 10000
    num_sweeps_to_train_with = 2
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size    
    lr = cntk_py.learning_rates_per_sample(0.02)
    input = variable((input_dim,), np.float32, needs_gradient=False, name="features")
    label = variable((num_output_classes,), np.float32, needs_gradient=False, name="labels")
    dev = cntk_py.DeviceDescriptor.cpudevice()     
    netout = fully_connected_classifier_net(input, num_output_classes, hidden_layers_dim, num_hidden_layers, dev, sigmoid)  
        
    ce = cross_entropy_with_softmax(netout.output(), label)
    pe = classification_error(netout.output(), label)
    ffnet = combine([ce, pe, netout], "classifier_model")      
    
    cm = create_mb_source(input_dim, num_output_classes, epoch_size)
       
    stream_infos = cm.stream_infos()    

    for si in stream_infos:
        if si.m_name == 'features':
            features_si = si
        elif si.m_name == 'labels':
            labels_si = si
    
    minibatch_size_limits = dict()    
    minibatch_size_limits[features_si] = (0,minibatch_size)
    minibatch_size_limits[labels_si] = (0,minibatch_size)
                          
    trainer = cntk_py.Trainer(ffnet, ce.output(), [cntk_py.sgdlearner(ffnet.parameters(), lr)])                   
    
    for i in range(0,int(num_minibatches_to_train)):        
        mb=cm.get_next_minibatch(minibatch_size_limits, dev)

        arguments = dict()
        arguments[input] = mb[features_si].m_data
        arguments[label] = mb[labels_si].m_data
            
        trainer.train_minibatch(arguments, dev)
        freq = 20
        if i % freq == 0: 
            training_loss = get_train_loss(trainer)                   
            print(str(i+freq) + ": " + str(training_loss))            
    #TODO: move the testing code into a separate test module ?
    assert np.allclose(training_loss, np.asarray([6.574318885803223]), atol=TOLERANCE_ABSOLUTE)
    
if __name__=='__main__':      
    test_ffnet()    
