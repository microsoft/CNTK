# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
import numpy as np
import cntk as C

from .converter import *

def _from_optimized_rnnstack(cudnn_rnn):
    '''
    converts cudnn optimized_rnnstack to non-cudnn functions to run in non-CUDA environment
    
    Args:
        cudnn_rnn: the optimized_rnnstack function that contains the parameters to be converted
    Returns:
        converted rnn function on GEMM based implementation that can be used on CPU
    '''

    if cudnn_rnn.root_function.op_name != 'OptimizedRNNStack':
        raise ValueError('unexpected cudnn_rnn.root_function.op_name value "%s"'%cudnn_rnn.root_function.op_name)
    
    cudnn_param = cudnn_rnn.parameters[0]
    rnn_name = cudnn_rnn.name
    input_var = cudnn_rnn.inputs[0]
    
    hidden_size = cudnn_rnn.root_function.attributes['hiddenSize']
    num_layers = cudnn_rnn.root_function.attributes['numLayers']
    bidirectional = cudnn_rnn.root_function.attributes['bidirectional']
    recurrent_op = cudnn_rnn.root_function.attributes['recurrentOp']

    if recurrent_op not in ['lstm', 'rnnReLU', 'rnnTanh']:
        raise ValueError('unsupported recurrent_op value "%s"'%recurrent_op)
    #note that cudnn GRU is different from standard GRU so no conversion unless creating a new type of GRU cell for CPU

    def _any_inferred(shape):
        return np.any([dim < 0 for dim in shape])
    
    if _any_inferred(cudnn_param.shape) or _any_inferred(input_var.shape):
        raise ValueError('parameter not initialized yet')

    input_size = input_var.shape[0] if len(input_var.shape) else 1
    
    num_gates = 1
    rnn_lambda = None
    if recurrent_op == 'lstm':
        num_gates = 4
        if bidirectional:
            rnn_lambda = lambda x, i : C.splice(C.layers.Recurrence(C.layers.LSTM(hidden_size, name=rnn_name+'_fw'+i))(x), C.layers.Recurrence(C.layers.LSTM(hidden_size, name=rnn_name+'_bw'+i), go_backwards=True)(x))
        else:
            rnn_lambda = lambda x, i : C.layers.Recurrence(C.layers.LSTM(hidden_size, name=rnn_name+"_"+i))(x)
    elif recurrent_op == 'rnnReLU' or recurrent_op == 'rnnTanh':
        num_gates = 1
        activation = C.relu if recurrent_op == 'rnnReLU' else C.tanh
        if bidirectional:
            rnn_lambda = lambda x, i : C.splice(C.layers.Recurrence(C.layers.RNNStep(hidden_size, activation=activation, name=rnn_name+'_fw'+i))(x), C.layers.Recurrence(C.layers.RNNStep(hidden_size, activation=activation, name=rnn_name+'_bw'+i), go_backwards=True)(x))
        else:
            rnn_lambda = lambda x, i : C.layers.Recurrence(C.layers.RNNStep(hidden_size, activation=activation, name=rnn_name+"_"+i))(x)

    noncudnn_func = rnn_lambda(input_var, '0')

    param = cudnn_param.value.reshape(-1)
    offset = 0
    multiplier = 2 if bidirectional else 1

    def _adjust_gate_order(W):
        if recurrent_op == 'lstm':
            if len(W.shape) == 2:
                i,f,m,o = np.hsplit(W, 4)
                return np.concatenate((i,m,f,o), axis=1)
            elif len(W.shape) == 1:
                i,f,m,o = np.split(W, 4)
                return np.concatenate((i,m,f,o))
            else:
                raise ValueError('LSTM parameter must have 1 or 2 dimensions')
        else:
            return W

    def _get_cudnn_rnn_weight_splitter(in_dim, h_dim):
        # for unidirectional, W, H
        # for bidirectional, fw_W, fw_H, bw_W, bw_H
        splitter = [in_dim*h_dim*num_gates, h_dim*h_dim*num_gates] * multiplier
        splitter = splitter[0:-1]
        return np.cumsum(splitter)

    def _get_cudnn_rnn_bias_splitter(h_dim):
        # for unidirectional, b1, b2
        # for bidirectional, fw_b1, fw_b2, bw_b1, bw_b2
        splitter = [h_dim*num_gates, h_dim*num_gates] * multiplier
        splitter = splitter[0:-1]
        return np.cumsum(splitter)

    offset = 0
    layer_input_size = input_size
    for layer in range(num_layers):
        layer_size = (layer_input_size + hidden_size) * hidden_size * num_gates * multiplier
        layer_param = param[offset:offset+layer_size]
        layer_name = str(layer)
        if bidirectional:
            fw_Wt, fw_Ht, bw_Wt, bw_Ht = np.split(layer_param, _get_cudnn_rnn_weight_splitter(layer_input_size, hidden_size))
            fw_cell = noncudnn_func.find_by_name(rnn_name+'_fw'+layer_name, -1)
            bw_cell = noncudnn_func.find_by_name(rnn_name+'_bw'+layer_name, -1)
            fw_cell.W.value = np.ascontiguousarray(_adjust_gate_order(fw_Wt.reshape(num_gates*hidden_size, -1).transpose()))
            fw_cell.H.value = np.ascontiguousarray(_adjust_gate_order(fw_Ht.reshape(num_gates*hidden_size, -1).transpose()))
            bw_cell.W.value = np.ascontiguousarray(_adjust_gate_order(bw_Wt.reshape(num_gates*hidden_size, -1).transpose()))
            bw_cell.H.value = np.ascontiguousarray(_adjust_gate_order(bw_Ht.reshape(num_gates*hidden_size, -1).transpose()))
        else:
            Wt, Ht = np.split(layer_param, _get_cudnn_rnn_weight_splitter(layer_input_size, hidden_size))
            cell = noncudnn_func.find_by_name(rnn_name+'_'+layer_name, -1)
            cell.W.value = np.ascontiguousarray(_adjust_gate_order(Wt.reshape(num_gates*hidden_size, -1).transpose()))
            cell.H.value = np.ascontiguousarray(_adjust_gate_order(Ht.reshape(num_gates*hidden_size, -1).transpose()))

        offset += layer_size
        layer_input_size = hidden_size * multiplier
        
        if layer != num_layers - 1:
            noncudnn_func = rnn_lambda(noncudnn_func.output, str(layer+1))

    for layer in range(num_layers):
        layer_size = 2 * hidden_size * num_gates * multiplier
        layer_param = param[offset:offset+layer_size]
        layer_name = str(layer)
        if bidirectional:
            fw_b1, fw_b2, bw_b1, bw_b2 = np.split(layer_param, _get_cudnn_rnn_bias_splitter(hidden_size))
            fw_cell = noncudnn_func.find_by_name(rnn_name+'_fw'+layer_name, -1)
            bw_cell = noncudnn_func.find_by_name(rnn_name+'_bw'+layer_name, -1)
            fw_cell.b.value = _adjust_gate_order(fw_b1 + fw_b2).reshape(-1)
            bw_cell.b.value = _adjust_gate_order(bw_b1 + bw_b2).reshape(-1)
        else:
            b1, b2 = np.split(layer_param, _get_cudnn_rnn_bias_splitter(hidden_size))
            cell = noncudnn_func.find_by_name(rnn_name+'_'+layer_name, -1)
            cell.b.value = _adjust_gate_order(b1 + b2).reshape(-1)
        offset += layer_size

    return noncudnn_func
    

def convert_optimized_rnnstack(cudnn_model):
    '''
    Converts model that contains cudnn optimized_rnnstack to use non-cudnn functions, 
    so it can be used in non-CUDA environment

    Args:
        cudnn_model: a model that contains optimized_rnnstacks
    Returns:
        Converted model on GEMM based implementation of rnn that can be used on CPU
    '''
    class CuDNNOptimizedRNNConverter:
        def __init__(self, model):
            self.filter = lambda x : type(x) == C.Function and x.root_function.op_name == 'OptimizedRNNStack'
            all_cudnn_rnns = C.logging.graph.depth_first_search(model, self.filter, depth=-1)
            unique_params = set([cudnn_rnn.parameters[0] for cudnn_rnn in all_cudnn_rnns])
            self.map_param_to_func = {p:None for p in unique_params}

        def converter(self, cudnn_rnn):
            param = cudnn_rnn.parameters[0]
            if self.map_param_to_func[param]:
                #shared parameter, clone
                converted = self.map_param_to_func[param][0].clone(C.CloneMethod.share, {self.map_param_to_func[param][1] : cudnn_rnn.inputs[0], self.map_param_to_func[param][2] : C.placeholder()})
            else:
                #unique or first parameter, convert
                converted = _from_optimized_rnnstack(cudnn_rnn)
                self.map_param_to_func[param] = (converted, cudnn_rnn.inputs[0], cudnn_rnn.output,)
            
            return converted

    optimizedRNNConverter = CuDNNOptimizedRNNConverter(cudnn_model)
    return convert(cudnn_model, optimizedRNNConverter.filter, optimizedRNNConverter.converter)
