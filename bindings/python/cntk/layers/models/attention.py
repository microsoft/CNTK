# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

'''
Standard attention model.
'''

from __future__ import division
import cntk as C
from cntk.ops.functions import Function
from cntk.default_options import default_options, get_default_override, default_override_or
from cntk.initializer import glorot_uniform
from ..layers import Dense, Label
from ..blocks import Stabilizer, _inject_name  # helpers
from ..sequence import PastValueWindow
from warnings import warn
#from .. import *


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

    compatible_attention_mode = True
    if attention_span is None:
        if attention_axis is not None:
            raise ValueError('attention_span cannot be None when attention_axis is not None')
        compatible_attention_mode = False
    elif attention_span <= 0:
        raise ValueError('attention_span must be a positive value')
    elif attention_axis is None:
        raise ValueError('attention_axis cannot be None when attention_span is not None')

    # model parameters
    with default_options(bias=False): # all the projections have no bias
        attn_proj_enc   = Stabilizer(enable_self_stabilization=enable_self_stabilization) >> Dense(attention_dim, init=init, input_rank=1) # projects input hidden state, keeping span axes intact
        attn_proj_dec   = Stabilizer(enable_self_stabilization=enable_self_stabilization) >> Dense(attention_dim, init=init, input_rank=1) # projects decoder hidden state, but keeping span and beam-search axes intact
        attn_proj_tanh  = Stabilizer(enable_self_stabilization=enable_self_stabilization) >> Dense(1            , init=init, input_rank=1) # projects tanh output, keeping span and beam-search axes intact
    attn_final_stab = Stabilizer(enable_self_stabilization=enable_self_stabilization)

    if compatible_attention_mode:
        warn('Specifying non-default values for attention_span and attention_axis has been deprecated since version 2.2. '
             'These arguments will be removed in the future.', DeprecationWarning, stacklevel=2)
        # old attention function
        @Function
        def old_attention(h_enc, h_dec):
            history_axis = h_dec # we use history_axis wherever we pass this only for the sake of passing its axis
            # TODO: pull this apart so that we can compute the encoder window only once and apply it to multiple decoders
            # --- encoder state window
            (h_enc, h_enc_valid) = PastValueWindow(attention_span, axis=attention_axis, go_backwards=go_backwards)(h_enc).outputs
            h_enc_proj = attn_proj_enc(h_enc)
            # window must be broadcast to every decoder time step
            h_enc_proj  = C.sequence.broadcast_as(h_enc_proj,  history_axis)
            h_enc_valid = C.sequence.broadcast_as(h_enc_valid, history_axis)
            # --- decoder state
            # project decoder hidden state
            h_dec_proj = attn_proj_dec(h_dec)
            tanh_out = C.tanh(h_dec_proj + h_enc_proj)  # (attention_span, attention_dim)
            u = attn_proj_tanh(tanh_out)              # (attention_span, 1)
            u_masked = u + (h_enc_valid - 1) * 50     # logzero-out the unused elements for the softmax denominator  TODO: use a less arbitrary number than 50
            attention_weights = C.softmax(u_masked, axis=attention_axis) #, name='attention_weights')
            attention_weights = Label('attention_weights')(attention_weights)
            # now take weighted sum over the encoder state vectors
            h_att = C.reduce_sum(C.element_times(C.sequence.broadcast_as(h_enc, history_axis), attention_weights), axis=attention_axis)
            h_att = attn_final_stab(h_att)
            return h_att

        return _inject_name(old_attention, name)
    else:
        # new attention function
        @Function
        def new_attention(encoder_hidden_state, decoder_hidden_state):
            # encode_hidden_state: [#, e] [h]
            # decoder_hidden_state: [#, d] [H]
            unpacked_encoder_hidden_state, valid_mask = C.sequence.unpack(encoder_hidden_state, padding_value=0).outputs
            # unpacked_encoder_hidden_state: [#] [*=e, h]
            # valid_mask: [#] [*=e]
            projected_encoder_hidden_state = C.sequence.broadcast_as(attn_proj_enc(unpacked_encoder_hidden_state), decoder_hidden_state)
            # projected_encoder_hidden_state: [#, d] [*=e, attention_dim]
            broadcast_valid_mask = C.sequence.broadcast_as(C.reshape(valid_mask, (1,), 1), decoder_hidden_state)
            # broadcast_valid_mask: [#, d] [*=e]
            projected_decoder_hidden_state = attn_proj_dec(decoder_hidden_state)
            # projected_decoder_hidden_state: [#, d] [attention_dim]
            tanh_output = C.tanh(projected_decoder_hidden_state + projected_encoder_hidden_state)
            # tanh_output: [#, d] [*=e, attention_dim]
            attention_logits = attn_proj_tanh(tanh_output)
            # attention_logits = [#, d] [*=e, 1]
            minus_inf = C.constant(-1e+30)
            masked_attention_logits = C.element_select(broadcast_valid_mask, attention_logits, minus_inf)
            # masked_attention_logits = [#, d] [*=e]
            attention_weights = C.softmax(masked_attention_logits, axis=0)
            attention_weights = Label('attention_weights')(attention_weights)
            # attention_weights = [#, d] [*=e]
            attended_encoder_hidden_state = C.reduce_sum(attention_weights * C.sequence.broadcast_as(unpacked_encoder_hidden_state, attention_weights), axis=0)
            # attended_encoder_hidden_state = [#, d] [1, h]
            output = attn_final_stab(C.reshape(attended_encoder_hidden_state, (), 0, 1))
            # output = [#, d], [h]
            return output

        return _inject_name(new_attention, name)
