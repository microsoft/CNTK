# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 00:23:44 2016

@author: wdarling
"""

import numpy as np
import sys
import os
from cntk import Trainer, Axis, save_model, load_model
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT, FULL_DATA_SWEEP
from cntk.device import cpu, set_default_device
from cntk.learner import momentum_sgd, momentum_schedule
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error, sequence, slice, past_value, future_value, element_select, alias, hardmax
from cntk.ops.functions import CloneMethod
from cntk.graph import find_nodes_by_name

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))
sys.path.append(os.path.join(abs_path, "..", "..", "bindings", "python"))
from examples.common.nn import LSTMP_component_with_self_stabilization, stabilize, linear_layer, print_training_progress

input_vocab_size = 69
label_vocab_size = 69

# model dimensions
input_vocab_dim  = input_vocab_size
label_vocab_dim  = label_vocab_size
hidden_dim = 256
num_layers = 1  

# helper function to find variables by name
# which is necessary when cloning or loading the model
def find_arg_by_name(name, expression):
    vars = [i for i in expression.arguments if i.name == name]
    assert len(vars) == 1
    return vars[0]

def create_model():
    
    # Source and target inputs to the model
    batch_axis = Axis.default_batch_axis()
    input_seq_axis = Axis('inputAxis')
    label_seq_axis = Axis('labelAxis')
    
    input_dynamic_axes = [batch_axis, input_seq_axis]
    raw_input = input_variable(
        shape=(input_vocab_dim), dynamic_axes=input_dynamic_axes, name='raw_input')
    
    label_dynamic_axes = [batch_axis, label_seq_axis]
    raw_labels = input_variable(
        shape=(label_vocab_dim), dynamic_axes=label_dynamic_axes, name='raw_labels')
    
    # Instantiate the sequence to sequence translation model
    input_sequence = raw_input
    
    # Drop the sentence start token from the label, for decoder training
    label_sequence = slice(raw_labels, label_seq_axis, 
                           1, 0, name='label_sequence') # <s> A B C </s> --> A B C </s>
    label_sentence_start = sequence.first(raw_labels)   # <s>
    
    # Setup primer for decoder
    is_first_label = sequence.is_first(label_sequence)  # 1 0 0 0 ...
    label_sentence_start_scattered = sequence.scatter(
        label_sentence_start, is_first_label)
    
    # Encoder
    encoder_output_h = stabilize(input_sequence)
    for i in range(0, num_layers):
        (encoder_output_h, encoder_output_c) = LSTMP_component_with_self_stabilization(
            encoder_output_h.output, hidden_dim, hidden_dim, future_value, future_value)
    
    # Prepare encoder output to be used in decoder
    thought_vector_h = sequence.first(encoder_output_h)
    thought_vector_c = sequence.first(encoder_output_c)
    
    thought_vector_broadcast_h = sequence.broadcast_as(
        thought_vector_h, label_sequence)
    thought_vector_broadcast_c = sequence.broadcast_as(
        thought_vector_c, label_sequence)
    
    # Decoder
    decoder_input = element_select(is_first_label, label_sentence_start_scattered, past_value(
        label_sequence))
    
    decoder_output_h = stabilize(decoder_input)
    for i in range(0, num_layers):
        if (i > 0):
            recurrence_hook_h = past_value
            recurrence_hook_c = past_value
        else:
            recurrence_hook_h = lambda operand: element_select(
                is_first_label, thought_vector_broadcast_h, past_value(operand))
            recurrence_hook_c = lambda operand: element_select(
                is_first_label, thought_vector_broadcast_c, past_value(operand))
    
        (decoder_output_h, encoder_output_c) = LSTMP_component_with_self_stabilization(
            decoder_output_h.output, hidden_dim, hidden_dim, recurrence_hook_h, recurrence_hook_c)
    
    # Softmax output layer
    model = linear_layer(stabilize(decoder_output_h), label_vocab_dim)
    
    return model