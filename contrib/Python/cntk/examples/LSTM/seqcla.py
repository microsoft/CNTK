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

# to be removed as they're added to the real API
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
    it_b = parameter((cell_dim, 1))
    it_h = times(parameter((cell_dim, output_dim)), prev_state_h)
    it_c = parameter((cell_dim, 1)) * prev_state_c        
    it = sigmoid(it_w + it_b + it_h + it_c)

    # applied to tanh of input    
    bit_w = times(parameter((cell_dim, input_dim)), x)
    bit_h = times(parameter((cell_dim, output_dim)), prev_state_h)
    bit_b = parameter((cell_dim, 1))
    bit = it * tanh(bit_w + (bit_h + bit_b))
        
    # forget-me-not gate (t)
    ft_w = times(parameter((cell_dim, input_dim)), x)
    ft_b = parameter((cell_dim, 1))
    ft_h = times(parameter((cell_dim, output_dim)), prev_state_h)
    ft_c = parameter((cell_dim, 1)) * prev_state_c        
    ft = sigmoid(ft_w + ft_b + ft_h + ft_c)

    # applied to cell(t-1)
    bft = ft * prev_state_c
        
    # c(t) = sum of both
    ct = bft + bit
        
    # output gate
    ot_w = times(parameter((cell_dim, input_dim)), x)
    ot_b = parameter((cell_dim, 1))
    ot_h = times(parameter((cell_dim, output_dim)), prev_state_h)
    ot_c = parameter((cell_dim, 1)) * prev_state_c        
    ot = sigmoid(ot_w + ot_b + ot_h + ot_c)
       
    # applied to tanh(cell(t))
    ht = ot * tanh(ct)
        
    # return cell value and hidden state
    return ct, ht

def seqcla():

    # LSTM params
    input_dim = 100
    output_dim = 128
    cell_dim = 128
    
    # model
    num_labels = 5
    vocab = 400001
    embed_dim = 100

    training_filename = "G:\BLIS\seqcla\sparse\Train_CoarseType.tsv.s"
    test_filename = "G:\BLIS\seqcla\sparse\Test_CoarseType.tsv.s"

    t = dynamic_axis()
    features = cntk1.SparseInput(vocab, dynamicAxis=t, var_name='features')    
    labels = input(num_labels, name='labels')
    
    #train_reader = CNTKTextFormatReader(training_filename)
    train_reader = CNTKTextFormatReader(test_filename)

    # setup embedding matrix
    embedding = parameter((embed_dim, vocab), learning_rate_multiplier=0.0, 
                          init='fromFile', init_from_file_path='G:\BLIS\seqcla\sparse\glove.6B.100D.txt.s')

    # get the vector representing the word
    sequence = times(embedding, features)
    
    # add an LSTM layer
    L = lstm_layer(output_dim, cell_dim, sequence, input_dim)
    
    # get only the last hidden state
    lst = Last(L)
    
    # add a softmax layer on top
    w = parameter((num_labels, output_dim))
    b = parameter((num_labels, 1))
    z = times(w, lst) + b
    
    ce = cntk1.CrossEntropyWithSoftmax(labels, z)
    ce.tag = "criterion"
    
    #my_sgd = SGDParams(epoch_size=0, minibatch_size=25, learning_ratesPerMB=0.1, max_epochs=3)    
    
    with Context('seqcla', clean_up=False) as ctx:
        ctx.eval(node = ce, 
                 input_map=train_reader.map(
                     features, alias='x', dim=vocab, format='Sparse').map(
                     labels, alias='y', dim=num_labels, format='Dense'))        
    
        #ctx.train(root_nodes=[ce,ev], optimizer=my_sgd, input_reader = {features:f_reader, labels:l_reader})                
        #result = ctx.test(input_reader = {features:f_reader, labels:l_reader})
        

if (__name__ == "__main__"):
    seqcla()