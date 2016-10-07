import numpy as np
import math
import sys
import os
from cntk import learning_rates_per_sample, DeviceDescriptor, Trainer, sgd
from cntk.ops import input_variable, squared_error, tanh, times, plus, parameter

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))
from examples.common.nn import print_training_progress

def create(x, layer_size, input_dim):
    encoding_matrices = []
    for dim in layer_size:
        try:
            shape = x.shape()
        except AttributeError:
            shape = x.output().shape()
        W = np.random.uniform(-1.0 / math.sqrt(input_dim), 1.0 /
                math.sqrt(input_dim), size=(shape[0], dim)).astype(np.float32)

        param_W = parameter(shape=(shape[0], dim), init=W)
        b = np.zeros([dim], dtype=np.float32)
        param_b = parameter(shape=(dim), init=b)
        encoding_matrices.append(W)
        output = tanh(plus(times(x, param_W) , param_b))
        x = output
    encoded_x = x
    
    layer_size.reverse()
    encoding_matrices.reverse()
    
    for i, dim in enumerate(layer_size[1:] + [input_dim]):

        W = np.transpose(encoding_matrices[i]).astype(np.float32)
        param_tr_W = parameter(shape=np.shape(W), init=W)
        b = np.zeros([dim], dtype=np.float32)
        param_tr_b = parameter(shape=(dim), init=b)
        output = tanh(plus(times(x, param_tr_W) , param_tr_b))
        x = output

    reconstructed_x = x

    return {
        'encoded': encoded_x,
        'decoded': reconstructed_x,
    }



if __name__=='__main__':
    target_device = DeviceDescriptor.cpu_device()
    DeviceDescriptor.set_default_device(target_device)
    input_dim = 8
    layer_size = [7,6,5]
    
    x = input_variable(shape=(input_dim,), data_type=np.float32)
    
    autoencoder = create(x, layer_size, input_dim)
    
    lr = learning_rates_per_sample(0.001)
    se = squared_error(autoencoder['decoded'], x)
    
    trainer = Trainer(autoencoder['decoded'], se, se, [sgd(autoencoder['decoded'].parameters(), lr)])
    
    # Get minibatches of training data and perform model training
    minibatch_size = 1
    num_samples_per_sweep = 1000
    num_sweeps_to_train_with = 5
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size
    training_progress_output_freq = 200
    for i in range(0, int(num_minibatches_to_train)):
        features  = np.random.normal(0, 0.2, size=(input_dim)).astype(float)
        trainer.train_minibatch({x : features})

        if i % training_progress_output_freq == 0:
            print(i, "Input: ", features)
            print(i, "Output: ", trainer.model.eval(arguments={x : [features]}))
            print_training_progress(trainer, i, training_progress_output_freq)

