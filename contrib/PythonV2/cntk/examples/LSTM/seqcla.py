# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:11:31 2016

@author: wdarling
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import cntk as C
from cntk.ops import cntk1
import numpy as np

cur_dir = os.path.dirname(__file__)

train_file = os.path.join(cur_dir, "Train_sparse.txt")
embedding_file = os.path.join(cur_dir, "embeddingmatrix.txt")

# =====================================================================================
# LSTM sequence classification
# =====================================================================================

# this class is a temporary stop-gap to use a BS macro that hasn't been fully 
# ported to the python API as of yet
class Last(C.ComputationNode):
    def __init__(self, x, op_name='BS.Sequences.Last', name=None):
        super(Last, self).__init__(params=['x'], op_name=op_name, name=name)
        self.x = x
        self.params_with_defaults = []
        self.rank = x.rank

def lstm_layer(output_dim, cell_dim, x, input_dim):    
        
    prev_state_h = C.past_value(0, 'lstm_state_h')
    prev_state_c = C.past_value(0, 'lstm_state_c')
        
    lstm_state_c, lstm_state_h = lstm_func(output_dim, cell_dim, x, input_dim, prev_state_h, prev_state_c)
    lstm_state_c.name = 'lstm_state_c'
    lstm_state_h.name = 'lstm_state_h'

    # return the last hidden state
    return Last(lstm_state_h)
    
# currently requires output_dim==cell_dim    
def lstm_func(output_dim, cell_dim, x, input_dim, prev_state_h, prev_state_c):
        
    # input gate (t)
    it_w = C.times(x,C.parameter((input_dim, cell_dim)))
    it_b = C.parameter((1,cell_dim))
    it_h = C.times(prev_state_h,C.parameter((output_dim, cell_dim)))
    it_c = C.parameter((1,cell_dim)) * prev_state_c        
    it = C.sigmoid((it_w + it_b + it_h + it_c), name='it')

    # applied to tanh of input    
    bit_w = C.times(x,C.parameter((input_dim,cell_dim)))
    bit_h = C.times(prev_state_h,C.parameter((output_dim,cell_dim)))
    bit_b = C.parameter((1,cell_dim))
    bit = it * C.tanh(bit_w + (bit_h + bit_b))
        
    # forget-me-not gate (t)
    ft_w = C.times(x, C.parameter((input_dim,cell_dim)))
    ft_b = C.parameter((1,cell_dim))
    ft_h = C.times(prev_state_h,C.parameter((output_dim,cell_dim)))
    ft_c = C.parameter((1,cell_dim)) * prev_state_c        
    ft = C.sigmoid((ft_w + ft_b + ft_h + ft_c), name='ft')

    # applied to cell(t-1)
    bft = ft * prev_state_c
        
    # c(t) = sum of both
    ct = bft + bit
        
    # output gate
    ot_w = C.times(x, C.parameter((input_dim,cell_dim)))
    ot_b = C.parameter((1,cell_dim))
    ot_h = C.times(prev_state_h,C.parameter((output_dim,cell_dim)))
    ot_c = C.parameter((1,cell_dim)) * prev_state_c        
    ot = C.sigmoid((ot_w + ot_b + ot_h + ot_c), name='ot')
       
    # applied to tanh(cell(t))
    ht = ot * C.tanh(ct)
        
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

    t = C.dynamic_axis(name='t')
    features = C.sparse_input(vocab, dynamic_axis=t, name='features')    
    labels = C.input(num_labels, name='labels')
   
    train_reader = C.CNTKTextFormatReader(train_file)

    # setup embedding matrix
    embedding = C.parameter((vocab, embed_dim), learning_rate_multiplier=0.0, 
                             init_from_file_path=embedding_file)

    # get the vector representing the word
    sequence = C.times(features, embedding, name='sequence')
    
    # add an LSTM layer
    L = lstm_layer(output_dim, cell_dim, sequence, input_dim)
    
    # add a softmax layer on top
    w = C.parameter((output_dim, num_labels), name='w')
    b = C.parameter((1,num_labels), name='b')
    z = C.times(L, w) + b
    z.name='z'
    z.tag = "output"
    
    # and reconcile the shared dynamic axis
    pred = C.reconcile_dynamic_axis(z, labels, name='pred')    
    
    ce = C.cross_entropy_with_softmax(labels, pred)
    ce.tag = "criterion"
    
    my_sgd = C.SGDParams(epoch_size=0, minibatch_size=10, learning_rates_per_mb=0.1, max_epochs=3)    
    
    with C.LocalExecutionContext('seqcla') as ctx:
        # train the model
        ctx.train(root_nodes=[ce], training_params=my_sgd, input_map=train_reader.map(
                  features, alias='x', dim=vocab, format='Sparse').map(
                  labels, alias='y', dim=num_labels, format='Dense'))        
        
        # write out the predictions
        ctx.write(input_map=train_reader.map(
                  features, alias='x', dim=vocab, format='Sparse').map(
                  labels, alias='y', dim=num_labels, format='Dense'))
                  
        # do some manual accuracy testing
        acc = calc_accuracy(train_file, ctx.output_filename_base)
        # and test for the same number...
        TOLERANCE_ABSOLUTE = 1E-02
        assert np.allclose(acc, 0.6006415396952687, atol=TOLERANCE_ABSOLUTE)

"""
Test the accuracy of the trained model.
"""
def calc_accuracy(test_file, output_filename_base):
    
    # load labels
    labels=[]
    with open(test_file, 'r') as f_in:      
        for l in f_in:
            dd = l.split('|')
            if len(dd) > 2:
                x = dd[2].strip().split(' ')[1:]
                labels.append(np.argmax(x))
                
    # load predicted answers
    predicted=[]
    with open(output_filename_base + ".z", 'r') as f_in:      
        for l in f_in:
            predicted.append(np.argmax(l.strip().split(' ')))
            
    correct = 0
    for i in range(len(labels)):
        if labels[i] == predicted[i]:
            correct += 1
    
    return float(correct) / float(len(labels))

"""
Test function so the test suite picks this up and runs it
"""
def test_lstm_sequence_classification():
    seqcla()

if (__name__ == "__main__"):
    seqcla()
