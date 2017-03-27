# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

'''
attention -- standard attention model
'''

from __future__ import division
from cntk.ops.functions import Function
from ..blocks import _inject_name # helpers
from .. import *


# AttentionModel block
def AttentionModel(attention_dim, attention_span=None, attention_axis=None,
                   init=default_override_or(glorot_uniform()),
                   go_backwards=default_override_or(False),
                   enable_self_stabilization=default_override_or(True), name=''):
    '''
    AttentionModel(attention_dim, attention_span=None, attention_axis=None, init=glorot_uniform(), go_backwards=False, enable_self_stabilization=True, name='')

    Layer factory function to create a function object that implements an attention model
    as described in Bahdanau, et al., "Neural machine translation by jointly learning to align and translate."
    '''

    init                      = get_default_override(AttentionModel, init=init)
    go_backwards              = get_default_override(AttentionModel, go_backwards=go_backwards)
    enable_self_stabilization = get_default_override(AttentionModel, enable_self_stabilization=enable_self_stabilization)

    # until CNTK can handle multiple nested dynamic loops, we require fixed windows and fake it
    if attention_span is None or attention_axis is None:
        raise NotImplementedError('AttentionModel currently requires a fixed attention_span and a static attention_axis to be specified')

    # model parameters
    with default_options(bias=False): # all the projections have no bias
        attn_proj_enc   = Stabilizer(enable_self_stabilization=enable_self_stabilization) >> Dense(attention_dim, init=init, input_rank=1) # projects input hidden state, keeping span axes intact
        attn_proj_dec   = Stabilizer(enable_self_stabilization=enable_self_stabilization) >> Dense(attention_dim, init=init, input_rank=1) # projects decoder hidden state, but keeping span and beam-search axes intact
        attn_proj_tanh  = Stabilizer(enable_self_stabilization=enable_self_stabilization) >> Dense(1            , init=init, input_rank=1) # projects tanh output, keeping span and beam-search axes intact
    attn_final_stab = Stabilizer(enable_self_stabilization=enable_self_stabilization)

    # attention function
    @Function
    def attention(h_enc, h_dec):
        history_axis = h_dec # we use history_axis wherever we pass this only for the sake of passing its axis
        # TODO: pull this apart so that we can compute the encoder window only once and apply it to multiple decoders
        # --- encoder state window
        (h_enc, h_enc_valid) = PastValueWindow(attention_span, axis=attention_axis, go_backwards=go_backwards)(h_enc).outputs
        h_enc_proj = attn_proj_enc(h_enc)
        # window must be broadcast to every decoder time step
        h_enc_proj  = sequence.broadcast_as(h_enc_proj,  history_axis)
        h_enc_valid = sequence.broadcast_as(h_enc_valid, history_axis)
        # --- decoder state
        # project decoder hidden state
        h_dec_proj = attn_proj_dec(h_dec)
        tanh_out = tanh(h_dec_proj + h_enc_proj)  # (attention_span, attention_dim)
        u = attn_proj_tanh(tanh_out)              # (attention_span, 1)
        u_masked = u + (h_enc_valid - 1) * 50     # logzero-out the unused elements for the softmax denominator  TODO: use a less arbitrary number than 50
        attention_weights = softmax(u_masked, axis=attention_axis) #, name='attention_weights')
        attention_weights = Label('attention_weights')(attention_weights)
        # now take weighted sum over the encoder state vectors
        h_att = reduce_sum(element_times(h_enc_proj, attention_weights), axis=attention_axis)
        h_att = attn_final_stab(h_att)
        return h_att

    return _inject_name(attention, name)
