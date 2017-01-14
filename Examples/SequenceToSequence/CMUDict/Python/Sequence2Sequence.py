# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import numpy as np
import os
from cntk import Trainer, Axis
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT, FULL_DATA_SWEEP
from cntk.learner import momentum_sgd, momentum_as_time_constant_schedule, learning_rate_schedule, UnitType
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error, sequence, past_value, future_value, \
                     element_select, alias, hardmax, placeholder_variable, combine, parameter, times
from cntk.ops.functions import CloneMethod, load_model, Function
from cntk.ops.sequence import broadcast_as
from cntk.graph import find_by_name, find_all_with_name
from cntk.layers import *
from cntk.initializer import glorot_uniform
from cntk.utils import log_number_of_parameters, ProgressPrinter
from attention import create_attention_augment_hook

########################
# variables and stuff  #
########################

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data")
MODEL_DIR = "."
TRAINING_DATA = "cmudict-0.7b.train-dev-20-21.ctf"
TESTING_DATA = "cmudict-0.7b.test.ctf"
VALIDATION_DATA = "tiny.ctf"
VOCAB_FILE = "cmudict-0.7b.mapping"

# model dimensions
input_vocab_dim  = 69
label_vocab_dim  = 69
hidden_dim = 128
num_layers = 2
attention_dim = 128
attention_span = 20
use_attention = False   #True  --BUGBUG (layers): not working for now due to has_aux
use_embedding = True
embedding_dim = 200
vocab = enumerate([w.strip() for w in open(os.path.join(DATA_DIR, VOCAB_FILE)).readlines()])

########################
# define the reader    #
########################

def create_reader(path, is_training):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        features = StreamDef(field='S0', shape=input_vocab_dim, is_sparse=True),
        labels   = StreamDef(field='S1', shape=label_vocab_dim, is_sparse=True)
    )), randomize = is_training, epoch_size = INFINITELY_REPEAT if is_training else FULL_DATA_SWEEP)

########################
# define the model     #
########################

inputAxis=Axis('inputAxis')
labelAxis=Axis('labelAxis')

def testit(r, with_labels=True):
    from cntk.blocks import Constant, Type
    if True:
    #try:
        r.dump()
        if with_labels:
            r.update_signature(Type(3, dynamic_axes=[Axis.default_batch_axis(), inputAxis]), 
                               Type(3, dynamic_axes=[Axis.default_batch_axis(), labelAxis]))
        else:
            r.update_signature(Type(3, dynamic_axes=[Axis.default_batch_axis(), inputAxis]))
        r.dump()
        if with_labels:
            res = r.eval({r.arguments[0]: [[[0.9, 0.7, 0.8]]], r.arguments[1]: [[[0, 1, 0]]]})
        else:
            res = r.eval({r.arguments[0]: [[[0.9, 0.7, 0.8]]]})
        print(res)
    #except Exception as e:
        print(e)
        r.dump()     # maybe some updates were already made?
        pass
    #input("hit enter")
    exit()

def LSTM_layer(input, output_dim, recurrence_hook_h=past_value, recurrence_hook_c=past_value, augment_input_hook=None, create_aux=False):
    aux_input = None
    has_aux   = False
    if augment_input_hook != None:
        has_aux = True
        if create_aux:
            aux_input = augment_input_hook(dh)
        else:
            aux_input = augment_input_hook

    dh = placeholder_variable()
    dc = placeholder_variable()
    LSTM_cell = LSTM(output_dim, enable_self_stabilization=True)
    if has_aux:    
        f_x_h_c = LSTM_cell(splice(input, aux_input), dh, dc)
    else:
        f_x_h_c = LSTM_cell(input, dh, dc)
    h_c = f_x_h_c.outputs

    h = recurrence_hook_h(h_c[0])
    c = recurrence_hook_c(h_c[1])

    replacements = { dh: h.output, dc: c.output }
    f_x_h_c.replace_placeholders(replacements)

    h = f_x_h_c.outputs[0]
    c = f_x_h_c.outputs[1]

    return combine([h]), combine([c]), aux_input

# Stabilizer >> num_layers * LSTM_layer
def LSTM_stack(input, num_layers, output_dim, recurrence_hook_h=past_value, recurrence_hook_c=past_value, augment_input_hook=None):

    create_aux = augment_input_hook != None

    # only the first layer should create an auxiliary input (the attention weights are shared amongs the layers)
    input = Stabilizer()(input)
    output_h, output_c, aux = LSTM_layer(input, output_dim, 
                                         recurrence_hook_h, recurrence_hook_c, augment_input_hook, create_aux)
    for layer_index in range(1, num_layers):
        (output_h, output_c, aux) = LSTM_layer(output_h.output, output_dim, recurrence_hook_h, recurrence_hook_c, aux, False)

    return (output_h, output_c)

# note: non-att s2s model in layer style:
# with default_options(go_backwards=True, enable_stabilization=Trye)
# encoder = For(range(2), lambda: Recurrence(LSTM(250))) >> Fold(LSTM(500))
# s0 = encoder(input)
# s0 |> decoder_model
# decoder_step = H' >> R2(G) >> R(F) >> D     # note: stateful w.r.t. n
#              :: (s0, history*) -> (z*)      # history = label_delayed or hard_max(z_delayed); s0 = encoder state
# decoder_model = Unfold(decoder_step, cond)  # for the case of decoding (training: decoder_step(s0, Delay(labels, initial_state='<s>'))
#               :: (s0) -> (z*)
# cond(x) = hard_max(x) == '</s>'
# !!!TODO!!!: implement numpy syntax as a wrapper for slice(), __getitem__(self, arg), if isinstance(arg, slice)...
#   then say cond(x) = hard_max(x)[index_of_EOS]  and  initial_state=embedding[index_of_BOS,:]
#   This is so cool!! But not working for time and batch axis :(
# where decoder has two LSTM layers F and G, and an embedding layer for output H
# In decoding, use input to guide output len using where(2 like input)
# Special processing:
#  - H' = (identity, H)   # pass through s0, which is the actual input
#  - R2(G) = R(G') with
#    G'(x, dh1, dc1) = G(x, dh1', dc1) with dh1' = output of H' if first step else Delay(h1)
#    R2(s0, history) = R(history, initial_state=s0)
#  - Unfold(f)(x) = x |> (E >> R(f))  until cond(output)
#    E broadcasts the initial state to the entire sequence, so that R can run over it
#    Better: This R should be a special recurrence. It really is an Unfold().
def create_model():
    # Embedding: (input*) --> (embedded_input*)
    # Right now assumes shared embedding and shared vocab size.
    embed = Embedding(embedding_dim) if use_embedding else identity

    # Encoder: (embedded_input*) --> (h0, c0)
    # Create multiple layers of LSTMs by passing the output of the i-th layer
    # to the (i+1)th layer as its input
    # This is the plain s2s encoder. The attention encoder will keep the entire sequence instead.
    # Note: We go_backwards.
    with default_options(enable_self_stabilization=True, go_backwards=True):
        encoder = Sequential([
            embed,
            Stabilizer(),
            For(range(num_layers-1), lambda:
                Recurrence(LSTM(hidden_dim))),
            Fold(LSTM(hidden_dim), return_full_state=True)
        ])

    # Decoder: (history*, h0, c0) --> z*
    # where history is one of these, delayed by 1 step and <s> prepended:
    #  - training: labels
    #  - testing:  its own output hardmax(z)
    with default_options(enable_self_stabilization=True):
        @Function
        def decoder(history, h0, c0):
            r = history
            r = Stabilizer()(r)
            for i in range(num_layers):
                r = RecurrenceFrom(LSTM(hidden_dim))(r, h0, c0) # :: r, h0, c0 -> h
            r = Stabilizer()(r)
            r = Dense(label_vocab_dim)(r)
            return r

    @Function
    def model(raw_input, raw_labels): # (input_sequence, decoder_history_sequence) --> (word_sequence)

        # Set up sequences...
        input_sequence = raw_input

        # Drop the sequence start token from the label, for decoder training
        label_sequence = sequence.slice(raw_labels, 1, 0, name='label_sequence') # <s> A B C </s> --> A B C </s>
        # label sequence is only used in training as the history --> move this out to trainer function

        #label_sequence_start = sequence.first(raw_labels)                        # <s>
        # ^^ This is a constant, no need for labels
        # TODO: replace label_sequence_start by one-hot vector with 1 at <s> position

        # Connect encoder and decoder
        encoder_output = encoder(input_sequence)

        # During training we use the ground truth as input to the decoder. During model execution,
        # we need to redirect the output of the network back in as the input to the decoder. We do this by
        # setting up a 'hook' whose output will be changed during model execution
        decoder_history_hook = alias(label_sequence, name='decoder_history_hook') # copy label_embedded

        # The input to the decoder always starts with the special label sequence start token.
        # Then, use the previous value of the label sequence (for training) or the output (for execution).
        # In decoding, 'decoder_history_hook' will be rewired to a feedback loop.
        # For that reason, we reconcile the dynamic axis, which for now can be done by adding a dummy zero to the input.
        # This will get hidden in Unfold(), and also streamlined.
        label_sequence_start = Constant(np.array([w=='<s>' for w in vocab], dtype=np.float32))
        decoder_input = delayed_value(embed(decoder_history_hook), initial_state=embed(label_sequence_start), dynamic_axes_like=label_sequence)
        # TODO: move the reconcile to the decoder creator.

        z = decoder(decoder_input, *encoder_output.outputs)
        return z

        # OLD CODE which I may still need later:
        # Parameters to the decoder stack depend on the model type (use attention or not)
        if use_attention:
            label_embedded = embed(label_sequence)
            augment_input_hook = create_attention_augment_hook(attention_dim, attention_span, 
                                                               label_embedded, encoder_output_h)
            recurrence_hook_h = past_value
            recurrence_hook_c = past_value
            decoder_output_h, _ = LSTM_stack(decoder_input, num_layers, hidden_dim, recurrence_hook_h, recurrence_hook_c, augment_input_hook)    
        else:
          if False:
            # William's original
            thought_vector_h, thought_vector_c = encoder_output.outputs
            # Here we broadcast the single-time-step thought vector along the dynamic axis of the decoder
            label_embedded = embed(label_sequence)
            thought_vector_broadcast_h = broadcast_as(thought_vector_h, label_embedded)
            thought_vector_broadcast_c = broadcast_as(thought_vector_c, label_embedded)
            augment_input_hook = None
            is_first_label = sequence.is_first(label_sequence)  # 1 0 0 0 ...
            def recurrence_hook_h(operand):
                return element_select(is_first_label, thought_vector_broadcast_h, past_value(operand))
            def recurrence_hook_c(operand):
                return element_select(is_first_label, thought_vector_broadcast_c, past_value(operand))
            decoder_output_h, _ = LSTM_stack(decoder_input, num_layers, hidden_dim, recurrence_hook_h, recurrence_hook_c, augment_input_hook)    
            z = Dense(label_vocab_dim) (Stabilizer()(decoder_output_h))    
          else:
            z = decoder(decoder_input, *encoder_output.outputs)

        return z

    return model

########################
# train action         #
########################

def train(train_reader, valid_reader, vocab, i2w, model, max_epochs, epoch_size):

    from cntk.blocks import Constant, Type

    model.update_signature(Type(input_vocab_dim, dynamic_axes=[Axis.default_batch_axis(), inputAxis]), 
                           Type(label_vocab_dim, dynamic_axes=[Axis.default_batch_axis(), labelAxis]))
                           #Type(label_vocab_dim, dynamic_axes=[Axis.default_batch_axis(), Axis('labelAxis')]))
    #model.dump('model after update')

    # do some hooks so that we can direct data to the right place
    label_sequence = find_by_name(model, 'label_sequence')
    decoder_history_hook = find_by_name(model, 'decoder_history_hook')

    # Criterion nodes
    # TODO: change to @Function to ensure parameter order (William seemed to have worked around it by naming them)
    arg_names = [arg.name for arg in model.arguments]

    ce = cross_entropy_with_softmax(model, label_sequence)
    errs = classification_error(model, label_sequence)

    arg_names = [arg.name for arg in ce.arguments]
    ce.dump()

    arg_names = [arg.name for arg in errs.arguments]

    # for this model during training we wire in a greedy decoder so that we can properly sample the validation data
    # This does not need to be done in training generally though
    def clone_and_hook():
        # network output for decoder history
        net_output = hardmax(model)

        # make a clone of the graph where the ground truth is replaced by the network output
        return model.clone(CloneMethod.share, {decoder_history_hook.output : net_output.output})

    # get a new model that uses the network output as input to the decoder
    decoder_output_model = clone_and_hook()

    # Instantiate the trainer object to drive the model training
    lr_per_sample = learning_rate_schedule(0.005, UnitType.sample)
    minibatch_size = 72
    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    clipping_threshold_per_sample = 2.3
    gradient_clipping_with_truncation = True
    learner = momentum_sgd(model.parameters,
                           lr_per_sample, momentum_time_constant,
                           gradient_clipping_threshold_per_sample=clipping_threshold_per_sample, 
                           gradient_clipping_with_truncation=gradient_clipping_with_truncation)
    trainer = Trainer(model, (ce, errs), learner)

    # Get minibatches of sequences to train with and perform model training
    i = 0
    mbs = 0
    sample_freq = 100

    # print out some useful training information
    log_number_of_parameters(model) ; print()
    progress_printer = ProgressPrinter(freq=30, tag='Training')

    # dummy for printing the input sequence below
    #from cntk import Function
    I = Constant(np.eye(input_vocab_dim))
    @Function
    def noop(input):
        return times(input, I)
    noop.update_signature(Type(input_vocab_dim, is_sparse=True))

    for epoch in range(max_epochs):

        while i < (epoch+1) * epoch_size:
            # get next minibatch of training data
            mb_train = train_reader.next_minibatch(minibatch_size)
            #trainer.train_minibatch({find_arg_by_name('raw_input' , model) : mb_train[train_reader.streams.features], 
            #                         find_arg_by_name('raw_labels', model) : mb_train[train_reader.streams.labels]})
            trainer.train_minibatch(mb_train[train_reader.streams.features], mb_train[train_reader.streams.labels])

            progress_printer.update_with_trainer(trainer, with_metric=True) # log progress

            # every N MBs evaluate on a test sequence to visually show how we're doing
            if mbs % sample_freq == 0:
                mb_valid = valid_reader.next_minibatch(minibatch_size)
                
                q = noop(mb_valid[valid_reader.streams.features])
                print_sequences(q, i2w)
                print(end=" -> ")
                
                # run an eval on the decoder output model (i.e. don't use the groundtruth)
                #e = decoder_output_model.eval({find_arg_by_name('raw_input' , decoder_output_model) : 
                #                               mb_valid[valid_reader.streams.features], 
                #                               find_arg_by_name('raw_labels', decoder_output_model) : 
                #                               mb_valid[valid_reader.streams.labels]})
                e = decoder_output_model(mb_valid[valid_reader.streams.features], mb_valid[valid_reader.streams.labels])
                print_sequences(e, i2w)

                # debugging attention (uncomment to print out current attention window on validation sequence)
                debug_attention(decoder_output_model, mb_valid, valid_reader)                

            i += mb_train[train_reader.streams.labels].num_samples
            mbs += 1

        # log a summary of the stats for the epoch
        progress_printer.epoch_summary(with_metric=True)
        
        # save the model every epoch
        model_filename = os.path.join(MODEL_DIR, "model_epoch%d.cmf" % epoch)
        
        # NOTE: we are saving the model with the greedy decoder wired-in. This is NOT necessary and in some
        # cases it would be better to save the model without the decoder to make it easier to wire-in a 
        # different decoder such as a beam search decoder. For now we save this one though so it's easy to 
        # load up and start using.
        decoder_output_model.save_model(model_filename)
        print("Saved model to '%s'" % model_filename)

########################
# write action         #
########################

def write(reader, model, vocab, i2w):
    
    minibatch_size = 1024
    progress_printer = ProgressPrinter(tag='Evaluation')
    
    while True:
        # get next minibatch of data
        mb = reader.next_minibatch(minibatch_size)
        if not mb:
            break

        # TODO: just use __call__() syntax
        e = model.eval({find_arg_by_name('raw_input' , model) : mb[reader.streams.features], 
                        find_arg_by_name('raw_labels', model) : mb[reader.streams.labels]})
        print_sequences(e, i2w)
        
        progress_printer.update(0, mb[reader.streams.labels].num_samples, None)

#######################
# test action         #
#######################

def test(reader, model, num_minibatches=None):
    
    # we use the test_minibatch() function so need to setup a trainer
    label_sequence = sequence.slice(find_arg_by_name('raw_labels', model), 1, 0)
    lr = learning_rate_schedule(0.007, UnitType.sample)
    momentum = momentum_as_time_constant_schedule(1100)
    ce = cross_entropy_with_softmax(model, label_sequence)
    errs = classification_error(model, label_sequence)
    trainer = Trainer(model, ce, errs, [momentum_sgd(model.parameters, lr, momentum)])

    test_minibatch_size = 1024

    # Get minibatches of sequences to test and perform testing
    i = 0
    total_error = 0.0
    while True:
        mb = reader.next_minibatch(test_minibatch_size)
        if not mb: break
        mb_error = trainer.test_minibatch({find_arg_by_name('raw_input' , model) : mb[reader.streams.features], 
                                           find_arg_by_name('raw_labels', model) : mb[reader.streams.labels]})
        total_error += mb_error
        i += 1
        
        if num_minibatches != None:
            if i == num_minibatches:
                break

    # and return the test error
    return total_error/i

########################
# interactive session  #
########################

def translate_string(input_string, model, vocab, i2w, show_attention=False, max_label_length=20):

    vdict = {vocab[i]:i for i in range(len(vocab))}
    w = [vdict["<s>"]] + [vdict[w] for w in input_string] + [vdict["</s>"]]
    
    features = np.zeros([len(w),len(vdict)], np.float32)
    for t in range(len(w)):
        features[t,w[t]] = 1    
    
    l = [vdict["<s>"]] + [0 for i in range(max_label_length)]
    labels = np.zeros([len(l),len(vdict)], np.float32)
    for t in range(len(l)):
        labels[t,l[t]] = 1
    
    #pred = model.eval({find_arg_by_name('raw_input' , model) : [features], 
    #                   find_arg_by_name('raw_labels', model) : [labels]})
    pred = model([features], [labels])
    
    # print out translation and stop at the sequence-end tag
    print(input_string, "->", end='')
    tlen = 1 # length of the output sequence
    prediction = np.argmax(pred, axis=2)[0]
    for i in prediction:
        phoneme = i2w[i]
        if phoneme == "</s>": break
        tlen += 1
        print(phoneme, end=' ')
    print()
    
    # show attention window (requires matplotlib, seaborn, and pandas)
    if show_attention:
    
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
    
        att = find_by_name(model, 'attention_weights')
        q = combine([model, att])
        output = q.forward({find_arg_by_name('raw_input' , model) : [features], 
                         find_arg_by_name('raw_labels', model) : [labels]},
                         att.outputs)
                         
        # set up the actual words/letters for the heatmap axis labels
        columns = [i2w[ww] for ww in prediction[:tlen]]
        index = [i2w[ww] for ww in w]
 
        att_key = list(output[1].keys())[0]
        att_value = output[1][att_key]
        
        # get the attention data up to the length of the output (subset of the full window)
        X = att_value[0,:tlen,:len(w)]
        dframe = pd.DataFrame(data=np.fliplr(X.T), columns=columns, index=index)
    
        # show the attention weight heatmap
        sns.heatmap(dframe)
        plt.show()

def interactive_session(model, vocab, i2w, show_attention=False):

    import sys

    while True:
        user_input = input("> ").upper()
        if user_input == "QUIT":
            break
        translate_string(user_input, model, vocab, i2w, show_attention=True)
        sys.stdout.flush()

########################
# helper functions     #
########################

def get_vocab(path):
    # get the vocab for printing output sequences in plaintext
    vocab = [w.strip() for w in open(path).readlines()]
    i2w = { i:w for i,w in enumerate(vocab) }
    w2i = { w:i for i,w in enumerate(vocab) }
    
    return (vocab, i2w, w2i)

# Given a vocab and tensor, print the output
def print_sequences(sequences, i2w):
    for s in sequences:
        print([i2w[np.argmax(w)] for w in s], sep=" ")

# helper function to find variables by name
# which is necessary when cloning or loading the model
def find_arg_by_name(name, expression):
    vars = [i for i in expression.arguments if i.name == name]
    assert len(vars) == 1
    return vars[0]

# to help debug the attention window
def debug_attention(model, mb, reader):
    att = find_by_name(model, 'attention_weights')
    if att:
        q = combine([model, att])
        output = q.forward({find_arg_by_name('raw_input' , model) : 
                             mb[reader.streams.features], 
                             find_arg_by_name('raw_labels', model) : 
                             mb[reader.streams.labels]},
                             att.outputs)

        att_key = list(output[1].keys())[0]
        att_value = output[1][att_key]
        print(att_value[0,0,:])

#############################
# main function boilerplate #
#############################

if __name__ == '__main__':

    from _cntk_py import set_computation_network_trace_level, set_fixed_random_seed, force_deterministic_algorithms
    set_fixed_random_seed(1)  # BUGBUG: has no effect at present  # TODO: remove debugging facilities once this all works

    # repro for name loss
    from cntk import plus, as_block
    from _cntk_py import InferredDimension
    arg = placeholder_variable()
    x = times(arg, parameter((InferredDimension,3), init=glorot_uniform()), name='x')
    x = sequence.first(x)
    sqr = x*x
    x1 = sqr.find_by_name('x')
    sqr2 = as_block(sqr, [(sqr.placeholders[0], placeholder_variable())], 'sqr')
    sqr2 = combine([sqr2])
    x2 = sqr2.find_by_name('x')

    stest = sqr2
    #stest.dump()
    stest = stest.replace_placeholders({stest.arguments[0]: input_variable(13)})
    #stest.dump()

    # hook up data
    train_reader = create_reader(os.path.join(DATA_DIR, TRAINING_DATA), True)
    valid_reader = create_reader(os.path.join(DATA_DIR, VALIDATION_DATA), True)
    vocab, i2w, w2i = get_vocab(os.path.join(DATA_DIR, VOCAB_FILE))

    # create inputs and create model
    #inputs = create_inputs()
    model = create_model()

    # train
    #try:
    train(train_reader, valid_reader, vocab, i2w, model, max_epochs=10, epoch_size=908241)
    #except:
    #    x = input("hit enter")

    # write
    #model = load_model("model_epoch0.cmf")
    #write(valid_reader, model, vocab, i2w)
    
    # test
    #model = load_model("model_epoch0.cmf")
    #test_reader = create_reader(os.path.join(DATA_DIR, TESTING_DATA), False)
    #test(test_reader, model)

    # test the model out in an interactive session
    #print('loading model...')
    #model_filename = "model_epoch0.cmf"
    #model = load_model(model_filename)
    #interactive_session(model, vocab, i2w, show_attention=True)
