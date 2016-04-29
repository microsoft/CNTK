# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:11:31 2016

@author: wdarling
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from cntk import *
from cntk.ops import *
from cntk.ops import cntk1


# =====================================================================================
# LSTM sequence classification
# =====================================================================================

# this class is a temporary stop-gap to use a BS macro that hasn't been fully 
# ported to the python API as of yet
class Last(ComputationNode):
    def __init__(self, x, name='BS.Sequences.Last', var_name=None):
        super(Last, self).__init__(params=['x'], name=name, var_name=var_name)
        self.x = x
        self.params_with_defaults = []


def lstm_layer(output_dim, cell_dim, x, input_dim):    
        
    prev_state_h = past_value(0, 'lstm_state_h')
    prev_state_c = past_value(0, 'lstm_state_c')
        
    lstm_state_c, lstm_state_h = lstm_func(output_dim, cell_dim, x, input_dim, prev_state_h, prev_state_c)
    lstm_state_c.var_name = 'lstm_state_c'
    lstm_state_h.var_name = 'lstm_state_h'

    # return the hidden state
    return lstm_state_h
    
# currently requires output_dim==cell_dim    
def lstm_func(output_dim, cell_dim, x, input_dim, prev_state_h, prev_state_c):
        
    # input gate (t)
    it_w = times(parameter((cell_dim, input_dim)), x)
    it_b = parameter((cell_dim))
    it_h = times(parameter((cell_dim, output_dim)), prev_state_h)
    it_c = parameter((cell_dim)) * prev_state_c        
    it = sigmoid((it_w + it_b + it_h + it_c), name='it')

    # applied to tanh of input    
    bit_w = times(parameter((cell_dim, input_dim)), x)
    bit_h = times(parameter((cell_dim, output_dim)), prev_state_h)
    bit_b = parameter((cell_dim))
    bit = it * tanh(bit_w + (bit_h + bit_b))
        
    # forget-me-not gate (t)
    ft_w = times(parameter((cell_dim, input_dim)), x)
    ft_b = parameter((cell_dim))
    ft_h = times(parameter((cell_dim, output_dim)), prev_state_h)
    ft_c = parameter((cell_dim)) * prev_state_c        
    ft = sigmoid((ft_w + ft_b + ft_h + ft_c), name='ft')

    # applied to cell(t-1)
    bft = ft * prev_state_c
        
    # c(t) = sum of both
    ct = bft + bit
        
    # output gate
    ot_w = times(parameter((cell_dim, input_dim)), x)
    ot_b = parameter((cell_dim))
    ot_h = times(parameter((cell_dim, output_dim)), prev_state_h)
    ot_c = parameter((cell_dim)) * prev_state_c        
    ot = sigmoid((ot_w + ot_b + ot_h + ot_c), name='ot')
       
    # applied to tanh(cell(t))
    ht = ot * tanh(ct)
        
    # return cell value and hidden state
    return ct, ht

"""
Train an LSTM-based sequence classification model.
"""
def seqcla():

    # LSTM params
    input_dim = 50
    output_dim = 128
    cell_dim = 128
    
    # model
    num_labels = 5
    vocab = 2000
    embed_dim = 50    

    t = dynamic_axis(name='t')
    # temporarily using cntk1 SpareInput because cntk2's Input() will simply allow sparse as a parameter
    features = cntk1.SparseInput(vocab, dynamicAxis=t, var_name='features')    
    labels = input(num_labels, name='labels')

    training_filename = "Train_sparse.txt"    
    train_reader = CNTKTextFormatReader(training_filename)    

    # setup embedding matrix
    embedding = parameter((embed_dim, vocab), learning_rate_multiplier=0.0, 
                          init='fromFile', init_from_file_path='embeddingmatrix.txt')

    # get the vector representing the word
    sequence = times(embedding, features, name='sequence')
    
    # add an LSTM layer
    L = lstm_layer(output_dim, cell_dim, sequence, input_dim)
    
    # get only the last hidden state
    lst = Last(L, var_name='lst')
    
    # add a softmax layer on top
    w = parameter((num_labels, output_dim), name='w')
    b = parameter((num_labels), name='b')
    z = plus(times(w, lst), b, name='z')
    z.tag = "output"
    
    # and reconcile the shared dynamic axis
    pred = reconcile_dynamic_axis(z, labels, name='pred')    
    
    ce = cntk1.CrossEntropyWithSoftmax(labels, pred)
    ce.tag = "criterion"
    
    my_sgd = SGDParams(epoch_size=0, minibatch_size=10, learning_rates_per_mb=0.1, max_epochs=10)    
    
    with LocalExecutionContext('seqcla', clean_up=False) as ctx:
        # train the model
        ctx.train(root_nodes=[ce], optimizer=my_sgd, input_map=train_reader.map(
                  features, alias='x', dim=vocab, format='Sparse').map(
                  labels, alias='y', dim=num_labels, format='Dense'))        
        
        # write out the predictions
        ctx.write(input_map=train_reader.map(
                  features, alias='x', dim=vocab, format='Sparse').map(
                  labels, alias='y', dim=num_labels, format='Dense'))
                  
        # do some manual accuracy testing
        calc_accuracy(training_filename, ctx.output_filename_base)

"""
Test the accuracy of the trained model.
"""
def calc_accuracy(test_file, output_filename_base):
    
    # load labels
    labels=[]
    with open(test_file, 'r', encoding='utf8') as f_in:      
        for l in f_in:
            dd = l.split('|')
            if len(dd) > 2:
                x = dd[2].strip().split(' ')[1:]
                labels.append(np.argmax(x))
                
    # load predicted answers
    predicted=[]
    with open(output_filename_base + ".z", 'r', encoding='utf8') as f_in:      
        for l in f_in:
            predicted.append(np.argmax(l.strip().split(' ')))
            
    correct = 0
    for i in range(len(labels)):
        if labels[i] == predicted[i]:
            correct += 1
    
    print(float(correct) / float(len(labels)))

if (__name__ == "__main__"):
    seqcla()