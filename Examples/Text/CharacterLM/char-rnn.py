# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import numpy as np
import os
from cntk import Trainer, Axis
from cntk.learner import momentum_sgd, momentum_as_time_constant_schedule, learning_rate_schedule, UnitType
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error
from cntk.ops.functions import load_model
from cntk.blocks import LSTM, Stabilizer
from cntk.layers import Recurrence, Dense
from cntk.models import LayerStack, Sequential
from cntk.utils import log_number_of_parameters, ProgressPrinter

# model hyperparameters
hidden_dim = 256
num_layers = 2
minibatch_size = 100 # also how much time we unroll the RNN for

# Get data
def get_data(p, minibatch_size, data, char_to_ix, vocab_dim):

    # the character LM predicts the next character so get sequences offset by 1
    xi = [char_to_ix[ch] for ch in data[p:p+minibatch_size]]
    yi = [char_to_ix[ch] for ch in data[p+1:p+minibatch_size+1]]
    
    # a slightly inefficient way to get one-hot vectors but fine for low vocab (like char-lm)
    X = np.eye(vocab_dim, dtype=np.float32)[xi]
    Y = np.eye(vocab_dim, dtype=np.float32)[yi]

    # return a list of numpy arrays for each of X (features) and Y (labels)
    return [X], [Y]

# Sample from the network
def sample(root, ix_to_char, vocab_dim, char_to_ix, prime_text='', use_hardmax=True, length=100, temperature=1.0):

    # temperature: T < 1 means smoother; T=1.0 means same; T > 1 means more peaked
    def apply_temp(p):
        # apply temperature
        p = np.power(p, (temperature))
        # renormalize and return
        return (p / np.sum(p))

    def sample_word(p):
        if use_hardmax:
            w = np.argmax(p, axis=2)[0,0]
        else:
            # normalize probabilities then take weighted sample
            p = np.exp(p) / np.sum(np.exp(p))            
            p = apply_temp(p)
            w = np.random.choice(range(vocab_dim), p=p.ravel())
        return w

    plen = 1
    prime = -1

    # start sequence with first input    
    x = np.zeros((1, vocab_dim), dtype=np.float32)    
    if prime_text != '':
        plen = len(prime_text)
        prime = char_to_ix[prime_text[0]]
    else:
        prime = np.random.choice(range(vocab_dim))
    x[0, prime] = 1
    arguments = ([x], [True])

    # setup a list for the output characters and add the initial prime text
    output = []
    output.append(prime)
    
    # loop through prime text
    for i in range(plen):            
        p = root.eval(arguments)        
        
        # reset
        x = np.zeros((1, vocab_dim), dtype=np.float32)
        if i < plen-1:
            idx = char_to_ix[prime_text[i+1]]
        else:
            idx = sample_word(p)

        output.append(idx)
        x[0, idx] = 1            
        arguments = ([x], [False])
    
    # loop through length of generated text, sampling along the way
    for i in range(length-plen):
        p = root.eval(arguments)
        idx = sample_word(p)
        output.append(idx)

        x = np.zeros((1, vocab_dim), dtype=np.float32)
        x[0, idx] = 1
        arguments = ([x], [False])

    # return output
    return ''.join([ix_to_char[c] for c in output])

def load_data_and_vocab(training_file):
    
    # load data
    rel_path = training_file
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)
    data = open(path, "r").read()
    chars = sorted(list(set(data)))
    data_size, vocab_size = len(data), len(chars)
    print('data has %d characters, %d unique.' % (data_size, vocab_size))
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }

    # write vocab for future use
    with open(path + ".vocab", "w") as ff:
        for c in chars:
            ff.write("%s\n" % c) if c != '\n' else ff.write("\n")
    
    return data, char_to_ix, ix_to_char, data_size, vocab_size

# Creates the model to train
def create_model(output_dim):
    
    return Sequential([        
        LayerStack(num_layers, lambda: 
                   Sequential([Stabilizer(), Recurrence(LSTM(hidden_dim), go_backwards=False)])),
        Dense(output_dim)
    ])

# Model inputs
def create_inputs(vocab_dim):
    batch_axis = Axis.default_batch_axis()
    input_seq_axis = Axis('inputAxis')

    input_dynamic_axes = [batch_axis, input_seq_axis]
    input_sequence = input_variable(shape=vocab_dim, dynamic_axes=input_dynamic_axes)
    label_sequence = input_variable(shape=vocab_dim, dynamic_axes=input_dynamic_axes)
    
    return input_sequence, label_sequence

# Creates and trains a character-level language model
def train_lm(training_file):

    # load the data and vocab
    data, char_to_ix, ix_to_char, data_size, vocab_dim = load_data_and_vocab(training_file)

    # Model the source and target inputs to the model
    input_sequence, label_sequence = create_inputs(vocab_dim)

    # create the model
    model = create_model(vocab_dim)
    
    # and apply it to the input sequence    
    z = model(input_sequence)

    # setup the criterions (loss and metric)
    ce = cross_entropy_with_softmax(z, label_sequence)
    errs = classification_error(z, label_sequence)

    # Instantiate the trainer object to drive the model training
    lr_per_sample = learning_rate_schedule(0.001, UnitType.sample)
    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    clipping_threshold_per_sample = 5.0
    gradient_clipping_with_truncation = True
    learner = momentum_sgd(z.parameters, lr_per_sample, momentum_time_constant,
                           gradient_clipping_threshold_per_sample=clipping_threshold_per_sample,
                           gradient_clipping_with_truncation=gradient_clipping_with_truncation)
    trainer = Trainer(z, ce, errs, learner)

    sample_freq = 1000
    epochs = 50
    minibatches_per_epoch = int((data_size / minibatch_size))
    minibatches = epochs * minibatches_per_epoch
    
    # print out some useful training information
    log_number_of_parameters(z) ; print()
    progress_printer = ProgressPrinter(freq=100, tag='Training')    
    
    e = 0
    p = 0
    for i in range(0, minibatches):

        if p + minibatch_size+1 >= data_size:
            p = 0
            e += 1
            model_filename = "models/shakespeare_epoch%d.dnn" % e
            z.save_model(model_filename)
            print("Saved model to '%s'" % model_filename)

        # get the data            
        features, labels = get_data(p, minibatch_size, data, char_to_ix, vocab_dim)

        # Specify the mapping of input variables in the model to actual minibatch data to be trained with
        # If it's the start of the data, we specify that we are looking at a new sequence (True)
        mask = [False] 
        if p == 0:
            mask = [True]
        arguments = ({input_sequence : features, label_sequence : labels}, mask)
        trainer.train_minibatch(arguments)

        progress_printer.update_with_trainer(trainer, with_metric=True) # log progress
        
        if i % sample_freq == 0:
            print(sample(z, ix_to_char, vocab_dim, char_to_ix))

        p += minibatch_size
        

def load_and_sample(model_filename, vocab_filename, prime_text='', use_hardmax=False, length=1000, temperature=1.0):
    
    # load the model
    model = load_model(model_filename)
    
    # load the vocab
    chars = [c[0] for c in open(vocab_filename).readlines()]
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
        
    output = sample(model, ix_to_char, len(chars), char_to_ix, prime_text=prime_text, use_hardmax=use_hardmax, length=length, temperature=temperature)
    
    ff = open('output.txt', 'w', encoding='utf-8')
    ff.write(output)
    ff.close()

if __name__=='__main__':    

    # train the LM    
    train_lm("data/tinyshakespeare.txt")

    # load and sample
    text = "T"
    load_and_sample("models/shakespeare_epoch0.dnn", "tinyshakespeare.txt.vocab", prime_text=text, use_hardmax=False, length=100, temperature=0.95)
