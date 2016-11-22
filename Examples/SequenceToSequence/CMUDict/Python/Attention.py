# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

########################################
# attention implementation for seq2seq #
########################################

from cntk.ops import plus, times, constant, past_value, reshape, sequence, splice, \
                     reduce_log_sum, exp, parameter, element_times, tanh, log, alias
from cntk.initializer import glorot_uniform
from cntk.blocks import Stabilizer

# create a past value window that returns two records: a value, shape=(N,dim), and a valid window, shape=(1,dim)
def past_value_window(N, input, axis=0):

    ones_like_input = plus(times(input, constant(0, shape=(input.shape[0],1))), constant(1, shape=(1)))
        
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
    value_a = splice(last_value, axis=axis)
    valid_a = splice(last_valid, axis=axis)

    # now reshape
    value = reshape(value_a, shape=(N, input.shape[0]))
    valid = reshape(valid_a, shape=(N, 1))

    # value[t] = value of t steps in the past; valid[t] = true if there was a value t steps in the past
    return value, valid

def my_softmax(z, axis):
    Z = reduce_log_sum(z, axis=axis) # reduce along axis
    P = exp(z - Z)
    
    return P

def create_attention_augment_hook(attention_dim, attention_span, decoder_dynamic_axis, encoder_outputH):

    stabilize = Stabilizer()

    # useful var
    encoder_output_dim = encoder_outputH.shape[0]

    # create the attention window
    (aw_value, aw_valid) = past_value_window(attention_span, encoder_outputH, axis=0)

    # setup the projection of the attention window to go into the tanh()
    def projected_attention_window_broadcast():
        W = parameter(shape=(attention_dim, encoder_output_dim), init=glorot_uniform())

        projected_value = sequence.broadcast_as(times(stabilize(element_times(aw_value, aw_valid)), W), 
                                                          decoder_dynamic_axis)
        value           = sequence.broadcast_as(aw_value, decoder_dynamic_axis)
        valid           = sequence.broadcast_as(aw_valid, decoder_dynamic_axis)

        # should be shape=(attention_span, attention_dim)
        return projected_value, value, valid

    # the function that gets passed to the LSTM function as the augment_input_hook parameter
    def augment_input_hook(input, prev_state):
        output_dim = prev_state.shape[0]
        W = parameter(shape=(output_dim, attention_dim), init=glorot_uniform())

        projectedH = times(stabilize(prev_state), W, output_rank=1)

        tanh_out = tanh(projectedH + projected_attention_window_broadcast()[0])  # (attention_span, attention_dim)
        
        # u = v * tanh(W1h + W2d)
        v = parameter(shape=(attention_dim, 1))
               
        u = times(stabilize(element_times(tanh_out, projected_attention_window_broadcast()[2])), v) # (attention_span, 1)
        u_valid = plus(u, log(projected_attention_window_broadcast()[2]), name='u_valid')

        attention_weights = alias(my_softmax(u_valid, 0), name='attention_weights')
        # the window should be shape=(attention_span, output_dim)
        weighted_attention_window = element_times(projected_attention_window_broadcast()[1], attention_weights, 
                                                  name='weighted_attention_window')

        ones = constant(value=1, shape=(attention_span))
        # weighted_attention_avg should be shape=(output_dim)
        weighted_attention_avg = times(ones, stabilize(weighted_attention_window), output_rank=1, 
                                       name='weighted_attention_avg')

        return weighted_attention_avg

    return augment_input_hook