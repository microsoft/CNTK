# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

########################################
# attention implementation for seq2seq #
########################################

from cntk.ops import times, constant, past_value, splice, softmax, reshape, \
                     parameter, element_times, tanh, log, alias
from cntk.ops.sequence import last, broadcast_as
from cntk.initializer import glorot_uniform
from cntk.blocks import Stabilizer, _INFERRED
from cntk.utils import _as_tuple

# Create a function which returns a static, maskable view for N past steps over a sequence along the given 'axis'.
# It returns two records: a value matrix, shape=(N,dim), and a valid window, shape=(1,dim)
def past_value_window(N, x, axis=0):
 
    # this is to create 1's along the same dynamic axis as `x`
    ones_like_input = times(x, constant(0, shape=(x.shape[0],1))) + 1
        
    last_value = []
    last_valid = []
    value = None
    valid = None

    for t in range(N):
        if t == 0:
            value = x
            valid = ones_like_input
        else:
            value = past_value(x, time_step=t)
            valid = past_value(ones_like_input, time_step=t)            
        
        last_value.append(last(value))
        last_valid.append(last(valid))

    # stack rows 'beside' each other, so axis=axis-2 (create a new static axis that doesn't exist)
    value = splice(last_value, axis=axis-2, name='value')
    valid = splice(last_valid, axis=axis-2, name='valid')

    # value[t] = value of t steps in the past; valid[t] = true if there was a value t steps in the past
    return (value, valid)

# This function creates a function to be passed to an RNN layer; it uses the input and hidden state of the 
# RNN to return an auxiliary input based on an attention over the hidden states.
def create_attention_augment_hook(attention_dim, attention_span, decoder_dynamic_axis, encoder_output_h):
   
    # inferred parameter shapes
    output_shape = _as_tuple(attention_dim)
    input_shape = _INFERRED

    # create the attention window function and get its return values
    aw_value, aw_valid = past_value_window(attention_span, encoder_output_h, axis=0)
    
    # setup the projection of the attention window to go into the tanh()
    def projected_attention_window_broadcast():
        W = parameter(shape=input_shape + output_shape, init=glorot_uniform())

        # We need to set up these 'broadcasted' versions so that these static-axis values can be properly 
        # broadcast to all of the steps along the dynamic axis that the decoder uses when we're calculating
        # the attention weights in the augment_input_hook function below
        projected_value = broadcast_as(times(Stabilizer()(element_times(aw_value, aw_valid)), W), 
                                                          decoder_dynamic_axis)
        value           = broadcast_as(aw_value, decoder_dynamic_axis)
        valid           = broadcast_as(aw_valid, decoder_dynamic_axis)

        # should be shape=(attention_span, attention_dim)
        return projected_value, value, valid

    # the function that gets passed to the LSTM function as the augment_input_hook parameter
    def augment_input_hook(prev_state):
        output_dim = prev_state.shape[0]
        W = parameter(shape=(output_dim, attention_dim), init=glorot_uniform())

        projectedH = times(Stabilizer()(prev_state), W, output_rank=1)        

        tanh_out = tanh(projectedH + projected_attention_window_broadcast()[0])  # (attention_span, attention_dim)
        
        # u = v * tanh(W1h + W2d)
        v = parameter(shape=(attention_dim, 1))
        
        u = times(Stabilizer()(element_times(tanh_out, projected_attention_window_broadcast()[2])), v) # (attention_span, 1)
                
        u_valid = u + log(projected_attention_window_broadcast()[2])

        # we do two reshapes (20,1)->(20) and then (20)->(20,1) so that we can use the built-in softmax()
        attention_weights = alias(softmax(reshape(u_valid, shape=(attention_span))), name='attention_weights')
        
        # the window should be shape=(attention_span, output_dim)
        weighted_attention_window = element_times(projected_attention_window_broadcast()[1], 
                                                  reshape(attention_weights, shape=(attention_span,1)), 
                                                  name='weighted_attention_window')

        ones = constant(value=1, shape=(attention_span))
        # weighted_attention_avg should be shape=(output_dim)
        weighted_attention_avg = times(ones, Stabilizer()(weighted_attention_window), output_rank=1, 
                                       name='weighted_attention_avg')

        return weighted_attention_avg

    return augment_input_hook