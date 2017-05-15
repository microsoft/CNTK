# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import tensorflow as tf
import numpy as np
from cntk.contrib import crosstalk as cstk

VariableType = 'Variable'
TrainableType = 'Trainable'
DictTrainableType = 'DictTrainable'

def find_trainable(name, scope=None):
    found = [tp for tp in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope) if name in tp.name]
    if len(found)==0:
        raise Exception('not found')
    elif len(found) > 1:
        raise Exception('more than 1 found')
    return found[0]

def trainable_setter(sess):
    def _set(p, raw_value, attr=None):
        if p.get_shape() != raw_value.shape:
            v = raw_value.reshape(p.get_shape())
        else:
            v = raw_value
        tf.assign(p, v).eval(session=sess)
    return _set
    
def trainable_getter(sess):
    def _get(p, attr=None):
        return p.eval(sess)
    return _get
    
def dict_trainable_setter(sess):
    def _set(td, raw_value, attr=None):
        if len(td) != len(raw_value):
            raise Exception('mismatch len')
        if td.keys() != raw_value.keys():
            raise Exception('mismatch keys')
        for k in td.keys():
            trainable_setter(sess)(td[k], raw_value[k])
    return _set

def dict_trainable_getter(sess):
    def _get(td, attr=None):
        return {k : trainable_getter(sess)(td[k]) for k in td.keys()}
    return _get

def variable_getter(sess, data):
    def _get(p, attr=None):
        return p.eval(data, session=sess)
    return _get

def conv2d_getter(sess):
    def _get(pd, attr):
        W = trainable_getter(sess)(pd.W)
        if pd.b:
            b = trainable_getter(sess)(pd.b)
        else:
            b = None
        return cstk.Conv2DArgs(W=W.transpose(2,0,1), b=b.reshape(attr.num_filters,))
    return _get

def conv2d_setter(sess):
    def _set(pd, raw_value, attr):
        trainable_setter(sess)(pd.W, raw_value.W.transpose(1,2,0))
        if pd.b:
            trainable_setter(sess)(pd.b, raw_value.b)
    return _set
    
def _adjust_forget_bias(all_bias, hidden_dim, forget_bias):
    i,m,f,o = np.split(all_bias, 4)
    f += forget_bias
    return np.concatenate((i,m,f,o))
    
def _rnn_trainable_in_scope(scope):
    if tf.VERSION.startswith('0.12'):
        fw_M=find_trainable('Matrix', scope=scope+'/FW')
        fw_b=find_trainable('Bias',   scope=scope+'/FW')
        bw_M=find_trainable('Matrix', scope=scope+'/BW')
        bw_b=find_trainable('Bias',   scope=scope+'/BW')
    elif tf.VERSION.startswith('1'):
        fw_M=find_trainable('weights', scope=scope+'/fw')
        fw_b=find_trainable('biases',   scope=scope+'/fw')
        bw_M=find_trainable('weights', scope=scope+'/bw')
        bw_b=find_trainable('biases',   scope=scope+'/bw')
    else:
        raise Exception('only supports 0.12.* and 1.*')

    return fw_M, fw_b, bw_M, bw_b

def rnn_getter(sess):
    def _get(scope, attr):
        if not attr.bidirectional:
            raise NotImplementedError()
        fw_M, fw_b, bw_M, bw_b = _rnn_trainable_in_scope(scope)
        fw_W, fw_H = np.split(trainable_getter(sess)(fw_M), [attr.input_dim])
        fw_b = _adjust_forget_bias(trainable_getter(sess)(fw_b), attr.hidden_dim, attr.forget_bias)
        bw_W, bw_H = np.split(trainable_getter(sess)(bw_M), [attr.input_dim])
        bw_b  = _adjust_forget_bias(trainable_getter(sess)(bw_b), attr.hidden_dim, attr.forget_bias)
        return cstk.RnnArgs(fw_W=fw_W, fw_H=fw_H, fw_b=fw_b, bw_W=bw_W, bw_H=bw_H, bw_b=bw_b)
    return _get

def rnn_setter(sess):
    def _set(scope, raw_value, attr):
        fw_M, fw_b, bw_M, bw_b = _rnn_trainable_in_scope(scope)
        if not attr.bidirectional:
            raise NotImplementedError()
        trainable_setter(sess)(fw_M, np.concatenate((raw_value.fw_W, raw_value.fw_H)))
        trainable_setter(sess)(fw_b, _adjust_forget_bias(raw_value.fw_b, attr.hidden_dim, -attr.forget_bias))
        trainable_setter(sess)(bw_M, np.concatenate((raw_value.bw_W, raw_value.bw_H)))
        trainable_setter(sess)(bw_b, _adjust_forget_bias(raw_value.bw_b, attr.hidden_dim, -attr.forget_bias))
    return _set

def embed_getter(sess):
    def _get(p, attr):
        map = {}
        value = trainable_getter(sess)(p)
        for i in range(attr.input_dim):
            map[attr.dict[i]] = value[i,:]
        return map
    return _get
    
def embed_setter(sess):
    def _set(p, raw_value, attr):
        out = [None]*attr.input_dim
        for w in raw_value.keys():
            out[attr.dict.index(w)] = raw_value[w] 
        trainable_setter(sess)(p, np.asarray(out))
    return _set

class TensorFlowCrosstalk(cstk.Crosstalk):
    '''
    TensorFlow implementation for crosstalk
    '''
    def __init__(self):
        super(TensorFlowCrosstalk, self).__init__()

    def set_data(self, sess, data):
        '''
        Set session and mapped data for setter/getters
        '''
        super(TensorFlowCrosstalk, self).register_funcs(TrainableType, setter=trainable_setter(sess), getter=trainable_getter(sess))
        super(TensorFlowCrosstalk, self).register_funcs(DictTrainableType, setter=dict_trainable_setter(sess), getter=dict_trainable_getter(sess))
        super(TensorFlowCrosstalk, self).register_funcs(VariableType, getter=variable_getter(sess, data))
        super(TensorFlowCrosstalk, self).register_funcs(cstk.Conv2DAttr, setter=conv2d_setter(sess), getter=conv2d_getter(sess))
        super(TensorFlowCrosstalk, self).register_funcs(cstk.RnnAttr, setter=rnn_setter(sess), getter=rnn_getter(sess))
        super(TensorFlowCrosstalk, self).register_funcs(cstk.EmbedAttr, setter=embed_setter(sess), getter=embed_getter(sess))
        
    def is_trainable(self, name):
        '''
        Check if variable with name is a trainable
        '''
        var_type = self.vars[name].type
        return var_type != VariableType
        
    def load_all_trainables(self):
        '''
        Load all trainables from files
        '''
        self.load([n for n in self.vars.keys() if self.is_trainable(n)])

    def save_all_trainables(self):
        '''
        Save all trainables to files
        '''
        self.save([n for n in self.vars.keys() if self.is_trainable(n)])

instance = TensorFlowCrosstalk()