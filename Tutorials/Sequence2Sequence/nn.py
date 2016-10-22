# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
from cntk.ops import *
from cntk.utils import get_train_eval_criterion, get_train_loss
from cntk.initializer import glorot_uniform


def linear_layer(input_var, output_dim):
    input_dim = input_var.shape[0]
    times_param = parameter(shape=(input_dim, output_dim), init=glorot_uniform())
    bias_param = parameter(shape=(output_dim), init=0)

    t = times(input_var, times_param)
    return bias_param + t


def fully_connected_layer(input, output_dim, nonlinearity):
    p = linear_layer(input, output_dim)
    return nonlinearity(p)

# Defines a multilayer feedforward classification model


def fully_connected_classifier_net(input, num_output_classes, hidden_layer_dim, num_hidden_layers, nonlinearity):
    r = fully_connected_layer(input, hidden_layer_dim, nonlinearity)
    for i in range(1, num_hidden_layers):
        r = fully_connected_layer(r, hidden_layer_dim, nonlinearity)

    return linear_layer(r, num_output_classes)


def conv_bn_layer(input, out_feature_map_count, kernel_width, kernel_height, h_stride, v_stride, w_scale, b_value, sc_value, bn_time_const):
    num_in_channels = input.shape[0]
    #TODO: use RandomNormal to initialize, needs to be exposed in the python api
    conv_params = parameter(shape=(out_feature_map_count, num_in_channels, kernel_height, kernel_width), init=glorot_uniform(output_rank=-1, filter_rank=2))
    conv_func = convolution(conv_params, input, (num_in_channels, v_stride, h_stride))

    #TODO: initialize using b_value and sc_value, needs to be exposed in the python api
    bias_params = parameter(shape=(out_feature_map_count), init=b_value)
    scale_params = parameter(shape=(out_feature_map_count), init=sc_value)
    running_mean = constant(0., (out_feature_map_count))
    running_invstd = constant(0., (out_feature_map_count))
    return batch_normalization(conv_func, scale_params, bias_params, running_mean, running_invstd, True, bn_time_const, 0.0, 0.000000001)


def conv_bn_relu_layer(input, out_feature_map_count, kernel_width, kernel_height, h_stride, v_stride, w_scale, b_value, sc_value, bn_time_const):
    conv_bn_function = conv_bn_layer(input, out_feature_map_count, kernel_width,
                                     kernel_height, h_stride, v_stride, w_scale, b_value, sc_value, bn_time_const)
    return relu(conv_bn_function)


def resnet_node2(input, out_feature_map_count, kernel_width, kernel_height, w_scale, b_value, sc_value, bn_time_const):
    c1 = conv_bn_relu_layer(input, out_feature_map_count, kernel_width,
                            kernel_height, 1, 1, w_scale, b_value, sc_value, bn_time_const)
    c2 = conv_bn_layer(c1, out_feature_map_count, kernel_width,
                       kernel_height, 1, 1, w_scale, b_value, sc_value, bn_time_const)
    p = c2 + input
    return relu(p)


def proj_layer(w_proj, input, h_stride, v_stride, b_value, sc_value, bn_time_const):
    num_in_channels = input.shape[0]
    conv_func = convolution(w_proj, input, (num_in_channels, v_stride, h_stride))
    out_feature_map_count = w_proj.shape[-1];
    #TODO: initialize using b_value and sc_value, needs to be exposed in the python api
    bias_params = parameter(shape=(out_feature_map_count), init=b_value)
    scale_params = parameter(shape=(out_feature_map_count), init=sc_value)
    running_mean = constant(0.0, (out_feature_map_count))
    running_invstd = constant(0.0, (out_feature_map_count))
    return batch_normalization(conv_func, scale_params, bias_params, running_mean, running_invstd, True, bn_time_const)


def resnet_node2_inc(input, out_feature_map_count, kernel_width, kernel_height, w_scale, b_value, sc_value, bn_time_const, w_proj):
    c1 = conv_bn_relu_layer(input, out_feature_map_count, kernel_width,
                            kernel_height, 2, 2, w_scale, b_value, sc_value, bn_time_const)
    c2 = conv_bn_layer(c1, out_feature_map_count, kernel_width,
                       kernel_height, 1, 1, w_scale, b_value, sc_value, bn_time_const)

    c_proj = proj_layer(w_proj, input, 2, 2, b_value, sc_value, bn_time_const)
    p = c2 + c_proj
    return relu(p)


def embedding(input, embedding_dim):
    input_dim = input.shape[0]

    embedding_parameters = parameter(shape=(input_dim, embedding_dim), init=glorot_uniform())
    return times(input, embedding_parameters)


def select_last(operand):
    return slice(operand, Axis.default_dynamic_axis(), -1, 0)


def stabilize(operand):
    scalar_constant = 4.0
    f = constant(scalar_constant)
    fInv = constant(1.0 / scalar_constant)

    beta = element_times(fInv, 
            log(1.0 + exp(element_times(f, parameter(init=0.99537863)))))
    return element_times(beta, operand)


def LSTMP_cell_with_self_stabilization(input, prev_output, prev_cell_state, aux, aux_dim):
    input_dim = input.shape[0]
    output_dim = prev_output.shape[0]
    cell_dim = prev_cell_state.shape[0]

    # setup parameters
    Wxo = parameter(shape=(input_dim, cell_dim), init=glorot_uniform())
    Wxi = parameter(shape=(input_dim, cell_dim), init=glorot_uniform())
    Wxf = parameter(shape=(input_dim, cell_dim), init=glorot_uniform())
    Wxc = parameter(shape=(input_dim, cell_dim), init=glorot_uniform())

    Aao = parameter(shape=(cell_dim, aux_dim), init=glorot_uniform())
    Aai = parameter(shape=(cell_dim, aux_dim), init=glorot_uniform())
    Aaf = parameter(shape=(cell_dim, aux_dim), init=glorot_uniform())
    Aac = parameter(shape=(cell_dim, aux_dim), init=glorot_uniform())

    Bo = parameter(shape=(cell_dim), init=0)
    Bc = parameter(shape=(cell_dim), init=0)
    Bi = parameter(shape=(cell_dim), init=0)
    Bf = parameter(shape=(cell_dim), init=0)

    Whi = parameter(shape=(output_dim, cell_dim), init=glorot_uniform())
    Wci = parameter(shape=(cell_dim), init=glorot_uniform())

    Whf = parameter(shape=(output_dim, cell_dim), init=glorot_uniform())
    Wcf = parameter(shape=(cell_dim), init=glorot_uniform())

    Who = parameter(shape=(output_dim, cell_dim), init=glorot_uniform())
    Wco = parameter(shape=(cell_dim), init=glorot_uniform())

    Whc = parameter(shape=(output_dim, cell_dim), init=glorot_uniform())

    Wmr = parameter(shape=(cell_dim, output_dim), init=glorot_uniform())

    # Stabilization by routing input through an extra scalar parameter
    sWxo = parameter(init=0)
    sWxi = parameter(init=0)
    sWxf = parameter(init=0)
    sWxc = parameter(init=0)

    sAao = parameter(init=0)
    sAai = parameter(init=0)
    sAaf = parameter(init=0)
    sAac = parameter(init=0)

    sWhi = parameter(init=0)
    sWci = parameter(init=0)

    sWhf = parameter(init=0)
    sWcf = parameter(init=0)
    sWho = parameter(init=0)
    sWco = parameter(init=0)
    sWhc = parameter(init=0)

    sWmr = parameter(init=0)

    expsWxo = exp(sWxo)
    expsWxi = exp(sWxi)
    expsWxf = exp(sWxf)
    expsWxc = exp(sWxc)

    expsAao = exp(sAao)
    expsAai = exp(sAai)
    expsAaf = exp(sAaf)
    expsAac = exp(sAac)

    expsWhi = exp(sWhi)
    expsWci = exp(sWci)

    expsWhf = exp(sWhf)
    expsWcf = exp(sWcf)
    expsWho = exp(sWho)
    expsWco = exp(sWco)
    expsWhc = exp(sWhc)

    expsWmr = exp(sWmr)

    Aaia = 0
    Aaca = 0
    Aafa = 0
    Aaoa = 0
    if aux != None:
        Aaia = times(element_times(expsAai, aux), Aai)
        Aaca = times(element_times(expsAac, aux), Aac)
        Aafa = times(element_times(expsAaf, aux), Aaf)
        Aaoa = times(element_times(expsAao, aux), Aao)

    Wxix = times(element_times(expsWxi, input), Wxi)
    Whidh = times(element_times(expsWhi, prev_output), Whi)
    Wcidc = element_times(Wci, element_times(expsWci, prev_cell_state))

    it = sigmoid(Wxix + Bi + Aaia + Whidh + Wcidc)
    Wxcx = times(element_times(expsWxc, input), Wxc)
    Whcdh = times(element_times(expsWhc, prev_output), Whc)
    bit = element_times(it, tanh(Wxcx + Bc + Aaca + Whcdh))
    Wxfx = times(element_times(expsWxf, input), Wxf)
    Whfdh = times(element_times(expsWhf, prev_output), Whf)
    Wcfdc = element_times(Wcf, element_times(expsWcf, prev_cell_state))

    ft = sigmoid(Wxfx + Bf + Aafa + Whfdh + Wcfdc)
    bft = element_times(ft, prev_cell_state)

    ct = bft + bit

    Wxox = times(element_times(expsWxo, input), Wxo)
    Whodh = times(element_times(expsWho, prev_output), Who)
    Wcoct = element_times(Wco, element_times(expsWco, ct))

    ot = sigmoid(Wxox + Bo + Aaoa + Whodh + Wcoct)

    mt = element_times(ot, tanh(ct))
    return (times(element_times(expsWmr, mt), Wmr), ct)

def ones_like(x):
    zero = constant(value=0, shape=(1))
    one  = constant(value=1, shape=(1))
    def zeroes_like(y):
        return element_times(y, zero)
    return plus(zeroes_like(x), one)

def past_value_window(N, input, axis=1):

    #ones_like_input = ones_like(input)
    ones_like_input = plus(times(input, constant(0, shape=(128, 1))), constant(1, shape=(1)))
    
    last_value=[]
    last_valid=[]
    value = None
    valid = None

    for t in range(N):
        if t == 0:
            value = input
            valid = ones_like_input
        else:
            value = past_value(input, time_step=t)
            valid = past_value(ones_like_input, time_step=t)     
        
        last_value.append(sequence.last(value))
        last_valid.append(sequence.last(valid))

    # can't get splice to stack rows 'beside' each other, so stack on top and then reshape...
    value_a = splice(last_value, axis=0)
    valid_a = splice(last_valid, axis=0)

    # workaround
    value = reshape(value_a, shape=(input.shape[0], N))
    #valid = reshape(valid_a, shape=(input.shape[0], N))
    valid = reshape(valid_a, shape=(1, N))

    # value[t] = value of t steps in the past; valid[t] = true if there was a value t steps in the past
    return value, valid

def my_softmax(z, axis):
    Z = reduce_log_sum(z, axis=axis) # reduce along axis
    P = exp(z - Z)
    return P

def create_attention_augment_hook(attention_dim, attention_span, decoder_dynamic_axis, encoder_outputH):

    # useful var
    encoder_output_dim = encoder_outputH.shape[0]

    # create the attention window
    (aw_value, aw_valid) = past_value_window(attention_span, encoder_outputH, axis=1)

    # setup the projection of the attention window to go into the tanh()
    def projected_attention_window_broadcast():
        W = parameter(shape=(attention_dim, encoder_output_dim), init=glorot_uniform())

        projected_value = sequence.broadcast_as(times(W, stabilize(element_times(aw_value, aw_valid))), 
                                                          decoder_dynamic_axis)
        value           = sequence.broadcast_as(aw_value, decoder_dynamic_axis)
        valid           = sequence.broadcast_as(aw_valid, decoder_dynamic_axis)

        return projected_value, value, valid

    # the function that gets passed to the LSTM function as the augment_input_hook parameter
    def augment_input_hook(input, prev_state):
        output_dim = prev_state.shape[0]
        W = parameter(shape=(attention_dim, output_dim), init=glorot_uniform())
        projectedH = times(W, stabilize(reshape(prev_state, shape=(output_dim,1))), output_rank=1)

        tanh_out = tanh(projectedH + projected_attention_window_broadcast()[0])  # (attention_dim, attention_span)

        # u = v * tanh(W1h + W2d)
        v = parameter(shape=(1, attention_dim))
        u = times(v, stabilize(element_times(tanh_out, projected_attention_window_broadcast()[2])))
        u_valid = plus(u, log(projected_attention_window_broadcast()[2]))

        import pdb; pdb.set_trace()

        attention_weights = my_softmax(u_valid, 1)
        weighted_attention_window = element_times(projected_attention_window_broadcast()[1], attention_weights)

        ones = constant(value=1, shape=(attention_span, 1))
        weighted_attention_avg = stabilize(times(weighted_attention_window, ones, output_rank=1))

        return reshape(weighted_attention_avg, shape=(output_dim))

    return augment_input_hook


def LSTMP_component_with_self_stabilization(input, output_dim, cell_dim, recurrence_hookH=past_value, recurrence_hookC=past_value, augment_input_hook=None, augment_input_dim=0):
    dh = placeholder_variable(
        shape=(output_dim), dynamic_axes=input.dynamic_axes)
    dc = placeholder_variable(
        shape=(cell_dim), dynamic_axes=input.dynamic_axes)

    aux_input = None
    if augment_input_hook != None:
        aux_input = augment_input_hook(input, dh)

    LSTMCell = LSTMP_cell_with_self_stabilization(input, dh, dc, aux_input, augment_input_dim)
    actualDh = recurrence_hookH(LSTMCell[0])
    actualDc = recurrence_hookC(LSTMCell[1])

    # Form the recurrence loop by replacing the dh and dc placeholders with
    # the actualDh and actualDc
    LSTMCell[0].replace_placeholders(
        {dh: actualDh.output, dc: actualDc.output})
    
    return (LSTMCell[0], LSTMCell[1])


def print_training_progress(trainer, mb, frequency):

    if mb % frequency == 0:
        training_loss = get_train_loss(trainer)
        eval_crit = get_train_eval_criterion(trainer)
        print("Minibatch: {}, Train Loss: {}, Train Evaluation Criterion: {}".format(
            mb, training_loss, eval_crit))
