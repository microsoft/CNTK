import numpy as np
import os
import sys
import cntk as C

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))
from examples.common.nn import print_training_progress

# Device set up
target_device = C.DeviceDescriptor.cpu_device()
C.DeviceDescriptor.set_default_device(target_device)

input_dim = 8
layers = [7,6]

shape = (input_dim,)
x = C.input_variable(shape=shape, data_type=np.float32)
Input = x

# Encoding layers
for dim in layers:
    W = np.random.normal(0, 1.0, size=(shape[0], dim)).astype(np.float32)
    param_W = C.parameter(init=W)
    b = np.random.normal(0, 1.0, size=[dim]).astype(np.float32)
    param_b = C.parameter(init=b)
    output = C.sigmoid(C.plus(C.times(x, param_W) , param_b))
    x = output
    try:
      shape = x.shape()
    except AttributeError:
      shape = x.output().shape()
encoded = x

try:
   prevDim = x.shape()[0]
except AttributeError:
   prevDim = x.output().shape()[0]

next_layers = layers[::-1][1:] + [input_dim]

# Decoding layers
for dim in next_layers:
    W = np.random.normal(0, 1.0, size=(prevDim, dim)).astype(np.float32)
    param_W = C.parameter(init=W)
    b = np.random.normal(0, 1.0, size=[dim]).astype(np.float32)
    param_b = C.parameter(init=b)
    output = C.sigmoid(C.plus(C.times(x, param_W) , param_b))
    x = output
    prevDim = dim
decoded = x

if __name__=='__main__':
    lr = C.learning_rates_per_sample(0.001)
    se = C.sqrt(C.reduce_mean(C.square(Input-decoded), axis=0))
    pe = C.sqrt(C.reduce_mean(C.square(Input-decoded), axis=0))

    trainer = C.Trainer(decoded, se, pe, [C.sgd(decoded.parameters(), lr)])
    
    # Get minibatches of training data and perform model training
    minibatch_size = 1
    num_samples_per_sweep = 1000
    num_sweeps_to_train_with = 5
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size
    training_progress_output_freq = 200

    for i in range(0, int(num_minibatches_to_train)):
        data = np.random.normal(0.6, 0.2, size=(input_dim))
        f = trainer.train_minibatch(arguments={Input : [data]})
        
        if i % training_progress_output_freq == 0:
            print(i, "Input: ", data)
            print(i, "Output: ", trainer.model().eval(arguments={Input : [data]}))
            print_training_progress(trainer, i, training_progress_output_freq)