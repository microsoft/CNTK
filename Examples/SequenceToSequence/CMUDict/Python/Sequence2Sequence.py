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
                     element_select, alias, hardmax, placeholder_variable, combine, parameter, times, plus
from cntk.ops.functions import CloneMethod, load_model, Function
from cntk.initializer import glorot_uniform
from cntk.utils import log_number_of_parameters, ProgressPrinter, debughelpers
from cntk.graph import find_by_name
from cntk.layers import *
from cntk.models.attention import *

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
attention_axis = -3
use_attention = True
use_embedding = True
embedding_dim = 200
vocab = ([w.strip() for w in open(os.path.join(DATA_DIR, VOCAB_FILE)).readlines()])
length_increase = 1.5

# sentence-start symbol as a constant
sentence_start = Constant(np.array([w=='<s>' for w in vocab], dtype=np.float32))
sentence_end_index = vocab.index('</s>')
# TODO: move these where they belong

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
    #from cntk.blocks import Constant, Type
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

# create the s2s model
def create_model(): # :: (history*, input*) -> logP(w)*
    # Embedding: (input*) --> (embedded_input*)
    # Right now assumes shared embedding and shared vocab size.
    embed = Embedding(embedding_dim, name='embed') if use_embedding else identity

    # Encoder: (input*) --> (h0, c0)
    # Create multiple layers of LSTMs by passing the output of the i-th layer
    # to the (i+1)th layer as its input
    # This is the plain s2s encoder. The attention encoder will keep the entire sequence instead.
    # Note: We go_backwards for the plain model, but forward for the attention model.
    with default_options(enable_self_stabilization=True, go_backwards=not use_attention):
        LastRecurrence = Fold if not use_attention else Recurrence
        encode = Sequential([
            embed,
            Stabilizer(),
            For(range(num_layers-1), lambda:
                Recurrence(LSTM(hidden_dim))),
            LastRecurrence(LSTM(hidden_dim), return_full_state=True),
            (Label('encoded_h'), Label('encoded_c')),
        ])

    # Decoder: (history*, input*) --> z*
    # where history is one of these, delayed by 1 step and <s> prepended:
    #  - training: labels
    #  - testing:  its own output hardmax(z)
    with default_options(enable_self_stabilization=True):
        # sub-layers
        stab_in = Stabilizer()
        rec_blocks = [LSTM(hidden_dim) for i in range(num_layers)]
        stab_out = Stabilizer()
        proj_out = Dense(label_vocab_dim, name='out_proj')
        # attention model
        if use_attention:
            attention_model = AttentionModel(attention_dim, attention_span, attention_axis, name='attention_model') # :: (h_enc*, h_dec) -> (h_aug_dec)
        # layer function
        # TODO: refactor such that it takes encoded_input (to be produced by model_train and model_greedy)
        @Function
        #def decode(history, x_last):
        #    input = x_last
        def decode(history, input):
            history_axis = history  # we use history_axis wherever we pass this only for the sake of passing its axis
            encoded_input = encode(input)
            r = history
            r = embed(r)
            r = stab_in(r)
            for i in range(num_layers):
                rec_block = rec_blocks[i]   # LSTM(hidden_dim)  # :: (x, dh, dc) -> (h, c)
                if use_attention:
                    if i == 0:
                        @Function
                        # TODO: undo x_last hack
                        def lstm_with_attention(dh, dc, x):
                            h_att = attention_model(encoded_input.outputs[0], dh)
                            x = splice(x, h_att)
                            r = rec_block(dh, dc, x)
                            (h, c) = r.outputs                   # BUGBUG: we need 'r', otherwise this will crash with an A/V
                            return (combine([h]), combine([c]))  # BUGBUG: we need combine(), otherwise this will crash with an A/V
                        r = Recurrence(lstm_with_attention)(r)
                    else:
                        r = Recurrence(rec_block)(r)
                else:
                    r = RecurrenceFrom(rec_block)(*encoded_input.outputs, r) # :: h0, c0, r -> h
            r = stab_out(r)
            r = proj_out(r)
            r = Label('out_proj_out')(r)
            return r

    return decode

########################
# train action         #
########################

def train(train_reader, valid_reader, vocab, i2w, s2smodel, max_epochs, epoch_size):

    #from cntk.blocks import Constant, Type

    # this is what we train here
    #s2smodel.update_signature(Type(label_vocab_dim, dynamic_axes=[Axis.default_batch_axis(), labelAxis]),
    #                          Type(input_vocab_dim, dynamic_axes=[Axis.default_batch_axis(), inputAxis]))
    # BUGBUG: fails with "Currently if an operand of a elementwise operation has any dynamic axes, those must match the dynamic axes of the other operands"
    #         Maybe also attributable to a parameter-order mix-up?
    # Need to think whether this makes sense. The axes are different for training and testing.

    # model used in training (history is known from labels)
    # note: the labels must not contain the initial <s>
    @Function
    #def model_train(input, x_last): # (input*, labels*) --> (word_logp*)
    #    labels = x_last
    def model_train(input, labels): # (input*, labels*) --> (word_logp*)

        # The input to the decoder always starts with the special label sequence start token.
        # Then, use the previous value of the label sequence (for training) or the output (for execution).
        # BUGBUG: This will fail with sparse input.
        past_labels = Delay(initial_state=sentence_start)(labels)
        return s2smodel(past_labels, input)

    # model used in (greedy) decoding (history is decoder's own output)
    @Function
    def model_greedy(input): # (input*) --> (word_sequence*)

        # Decoding is an unfold() operation starting from sentence_start.
        # We must transform s2smodel (history*, input* -> word_logp*) into a generator (history* -> output*)
        # which holds 'input' in its closure.
        unfold = UnfoldFrom(lambda history: s2smodel(history, input) >> hardmax,
                            #until_predicate=lambda w: w[...,sentence_end_index],  # stop once sentence_end_index was max-scoring output
                            # BUGBUG: causes some strange MBLayout error
                            length_increase=length_increase, initial_state=sentence_start)
        return unfold(dynamic_axes_like=input)

    try:
      model_greedy.update_signature(Type(input_vocab_dim, dynamic_axes=[Axis.default_batch_axis(), inputAxis]))
    except:
      debughelpers.dump_function(model_greedy, 'model_greedy')
      raise
    #model_greedy.dump()
    from cntk.graph import output_function_graph
    #output_function_graph(model_train, pdf_file_path=os.path.join(MODEL_DIR, "model") + '.pdf', scale=1)

    @Function
    def criterion(input, labels):
        #labels = x_last
        # criterion function must drop the <s> from the labels
        postprocessed_labels = sequence.slice(labels, 1, 0) # <s> A B C </s> --> A B C </s>
        z = model_train(input, postprocessed_labels)
        ce   = cross_entropy_with_softmax(z, postprocessed_labels)
        errs = classification_error      (z, postprocessed_labels)
        return (Function.NamedOutput(loss=ce), Function.NamedOutput(metric=errs))
    try:
      #criterion.dump()
      criterion.update_signature(input=Type(input_vocab_dim, dynamic_axes=[Axis.default_batch_axis(), inputAxis]), 
                               #x_last=Type(label_vocab_dim, dynamic_axes=[Axis.default_batch_axis(), labelAxis]))
                               labels=Type(label_vocab_dim, dynamic_axes=[Axis.default_batch_axis(), labelAxis]))
    except:
      #criterion.dump()
      raise
    debughelpers.dump_signature(criterion)
    #debughelpers.dump_function(criterion)
    output_function_graph(criterion, pdf_file_path=os.path.join(MODEL_DIR, "model") + '.pdf', scale=1)

    # for this model during training we wire in a greedy decoder so that we can properly sample the validation data
    # This does not need to be done in training generally though
    # Instantiate the trainer object to drive the model training
    lr_per_sample = learning_rate_schedule(0.005, UnitType.sample)
    minibatch_size = 72
    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    clipping_threshold_per_sample = 2.3
    gradient_clipping_with_truncation = True
    learner = momentum_sgd(model_train.parameters,
                           lr_per_sample, momentum_time_constant,
                           gradient_clipping_threshold_per_sample=clipping_threshold_per_sample, 
                           gradient_clipping_with_truncation=gradient_clipping_with_truncation)
    trainer = Trainer(None, criterion, learner)

    # Get minibatches of sequences to train with and perform model training
    i = 0
    mbs = 0
    sample_freq = 100

    # print out some useful training information
    log_number_of_parameters(model_train) ; print()
    progress_printer = ProgressPrinter(freq=30, tag='Training')

    # dummy for printing the input sequence below. Currently needed because input is sparse.
    I = Constant(np.eye(input_vocab_dim))
    @Function
    def no_op(input):
        return times(input, I)
    no_op.update_signature(Type(input_vocab_dim, is_sparse=True))

    for epoch in range(max_epochs):

        while i < (epoch+1) * epoch_size:
            # get next minibatch of training data
            mb_train = train_reader.next_minibatch(minibatch_size)
            trainer.train_minibatch(mb_train[train_reader.streams.features], mb_train[train_reader.streams.labels])

            progress_printer.update_with_trainer(trainer, with_metric=True) # log progress

            # every N MBs evaluate on a test sequence to visually show how we're doing
            if mbs % sample_freq == 0:
                mb_valid = valid_reader.next_minibatch(1)

                # run an eval on the decoder output model (i.e. don't use the groundtruth)
                e = model_greedy(mb_valid[valid_reader.streams.features])
                print_sequences(no_op(mb_valid[valid_reader.streams.features]), i2w)
                print("->")
                print_sequences(e, i2w)

                # debugging attention (uncomment to print out current attention window on validation sequence)
                if use_attention:
                    debug_attention(model_greedy, mb_valid[valid_reader.streams.features])

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
        model_greedy.save_model(model_filename)
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
    momentum = momentum_as_time_constant_schedule(1100) # BUGBUG: use Evaluator

    # BUGBUG: Must do the same as in train(), drop the first token
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
def debug_attention(model, input):
    #q = combine([model, model.attention_model.attention_weights, model.attention_model.u_masked, model.attention_model.h_enc_valid])
    #words, p, u, v = q(input)
    #W = model.out_proj.W
    #print(W.is_parameter)
    #Wv = W.value   # BUGBUG: fails. is_parameter, but is not Parameter
    #print(model.out_proj_out.W)
    q = combine([model, model.attention_model.attention_weights])
    words, p = q(input)
    len = words.shape[attention_axis-1]
    span = 7 #attention_span  #7 # test sentence is 7 tokens long
    p_sq = np.squeeze(p[0,:len,:span,0,:]) # (batch, len, attention_span, 1, vector_dim)
    #u_sq = np.squeeze(u[0,:len,:span,0,:]) # (batch, len, attention_span, 1, vector_dim)
    #v_sq = np.squeeze(v[0,:len,:span,0,:]) # (batch, len, attention_span, 1, vector_dim)
    #print(p_sq.shape, p_sq, u_sq, v_sq)
    opts = np.get_printoptions()
    np.set_printoptions(precision=5)
    print(p_sq)
    np.set_printoptions(**opts)

#############################
# main function boilerplate #
#############################

### BEGIN UNRELATED TEST
# a test I did for porting a Keras model for Xinying Song

# Configurations

#def merge_helper(mode, layer_list):
#    """
#    # Args:
#        mode (str):
#        layer_list (list[layer]): OK to have None layers
#    # Returns:
#        layer or None (if layer_list is empty)
#    """
#    layer_list = [item for item in layer_list if item is not None]
#    if len(layer_list) >= 2:
#        return Merge(mode=mode)(layer_list)
#    elif len(layer_list) == 1:
#        return layer_list[0]
#    else:
#        return None
#def create_regularizer():
#    if reg_l1 > 1e-7 and reg_l2 > 1e-7:
#        return regularizers.WeightRegularizer(l1=reg_l1, l2=reg_l2)
#    if reg_l1 > 1e-7:
#        return regularizers.WeightRegularizer(l1=reg_l1)
#    if reg_l2 > 1e-7:
#        return regularizers.WeightRegularizer(l2=reg_l2)
#    return None

def build_model_xy(model_save_path=None):

    rnn_type = 'LSTM'
    #rnn_type = 'GRU'
    dnn_hid_size_list = [32,32,32,32,32]
    rnn_hid_size_list = [128, 128]
    batch_size = 128
    reg_l1 = 0
    reg_l2 = 0
    dropout = 0
    batch_normalization = True
    residual_dnn = False
    skip_connection = True
    batch_size_predict = 1024
    # in cross-validation, we test up to this number and see the upper bound of RNN capability
    # then in train/test, we use the best epoch number obtained by cross-validation
    nb_epoch = 30
    patience = 5 # no use now because we don't do early stopping for now
    if rnn_type == 'GRU':
        rnn_cell = GRU
    elif rnn_type == "LSTM":
        rnn_cell = LSTM
    else:
       assert False

    # fake some environment
    rnn_dataset_dict=Record(train=Record(multiple_input=Record(
        rnn = Record(shape=(100,-123,250)),   # (batch, max_len, dim)   max_len not needed/used by CNTK
        dnn = Record(shape=(100,300))         # (batch, dim)
    )))
    model_flags = Record(  # True if input is to be included
        rnn=True,
        dnn=True
    )
    logger = Record(
        info = lambda *args: print(*args)
    )

    # Build model
    multiple_input = rnn_dataset_dict['train'].multiple_input
    for key in multiple_input.keys():
        logger.info((key, model_flags[key], multiple_input[key].shape))
    # construct inputs
    rnn_inputs_dict_by_length = {} # a map from length to rnn_inputs
    rnn_input_total_dim_by_length = {}
    rnn_inputs = [] # for feeding graph input only
    
    dnn_inputs = []
    dnn_input_total_dim = 0
    for key in multiple_input.keys():
        if not key in model_flags or not model_flags[key]:
            logger.info("Skipping input {0}".format(key))
            continue
        logger.info("Created input {0}".format(key))
        flds = key.split('-')
        #inp = Input(shape=multiple_input[key].shape[1:], name=key)
        if flds[0] == 'rnn':
            inp = Input(shape=multiple_input[key].shape[2:], name=key) # no explicit length dimension in CNTK
            rnn_inputs.append(inp) # for graph input only
            timesteps = multiple_input[key].shape[1]
            if timesteps not in rnn_inputs_dict_by_length:
                rnn_inputs_dict_by_length[timesteps] = ([], [])
            rnn_inputs_dict_by_length[timesteps][0].append(inp)
            if timesteps not in rnn_input_total_dim_by_length:
                rnn_input_total_dim_by_length[timesteps] = 0
            rnn_input_total_dim_by_length[timesteps] += multiple_input[key].shape[-1] 
        elif flds[0] == 'dnn':
            #inp = Input(shape=multiple_input[key].shape[1:], name=key)
            inp = Input(shape=multiple_input[key].shape[1:], name=key, dynamic_axes=[Axis.default_batch_axis()]) # dnn input has no sequence dimension
            dnn_input_total_dim += multiple_input[key].shape[-1]
            dnn_inputs.append(inp)
    
    # construct RNN layers
    rnn_hid_list = []
    # notes: rnn_inputs_dict_by_length.keys() currently has only one element
    for timesteps in rnn_inputs_dict_by_length.keys():
        #rnn_final_input = merge_helper('concat', 
        rnn_final_input = splice(*
                rnn_inputs_dict_by_length[timesteps][0])
        if rnn_final_input is not None:
            #rnn_hid = Masking(mask_value=0.)(rnn_final_input)
            rnn_hid = rnn_final_input
            rnn_output_dim = rnn_hid_size_list[-1]
            rnn_skip_connections = []
            prev_output_dim = rnn_input_total_dim_by_length[timesteps]
            if skip_connection:
                #tmp_out = ZeroMaskedEntries()(rnn_hid)
                tmp_out = rnn_hid
                #tmp_out = Lambda(lambda x: x[:,-1,:], output_shape=lambda input_shape: (input_shape[0], input_shape[2]))(tmp_out)
                tmp_out = sequence.last(tmp_out)
                if prev_output_dim != rnn_output_dim:
                    tmp_out = Dense(rnn_output_dim #,
                                            #W_regularizer=create_regularizer(),  # CNTK regularizers work differently
                                            #b_regularizer=create_regularizer(),
                                           )(tmp_out)
                rnn_skip_connections.append(tmp_out)
            for (depth, hid_size) in enumerate(rnn_hid_size_list):
                #rnn_hid = rnn_cell(hid_size,
                Recurrence_f = Recurrence if depth < len(rnn_hid_size_list)-1 else Fold # CNKT uses two different functions for return_sequences
                cell = rnn_cell(hid_size) # rnn_cell is the layer factory; create the layer so that we can know len(cell.outputs)
                rnn_hid = Recurrence_f(cell >> (Dropout(dropout),) * len(cell.outputs) # apply Dropout to all outputs
                                   #return_sequences=(True if depth < len(rnn_hid_size_list)-1 else False),
                                   #W_regularizer=create_regularizer(), # CNTK regularizers work differently
                                   #U_regularizer=create_regularizer(),
                                   #b_regularizer=create_regularizer(),
                                   #dropout_W=dropout,
                                   #dropout_U=dropout
                         )(rnn_hid)
                pre_output_dim = hid_size
                if skip_connection:
                    if depth == len(rnn_hid_size_list)-1:
                        tmp_out = rnn_hid
                    else:
                        #tmp_out = ZeroMaskedEntries()(rnn_hid)
                        tmp_out = rnn_hid
                        #tmp_out = Lambda(lambda x: x[:,-1,:], output_shape=lambda input_shape: (input_shape[0], input_shape[2]))(tmp_out)
                        tmp_out = sequence.last(tmp_out)
                    if prev_output_dim != rnn_output_dim:
                        tmp_out = Dense(rnn_output_dim #,
                                                #W_regularizer=create_regularizer(),  # CNTK regularizers work differently
                                                #b_regularizer=create_regularizer(),
                                               )(tmp_out)
                    rnn_skip_connections.append(tmp_out)
            if skip_connection:
                #rnn_hid = merge_helper('sum', rnn_skip_connections)
                rnn_hid = plus(*rnn_skip_connections)
            else:
                pass
        else:
            rnn_hid = None
        rnn_hid_list.append(rnn_hid)
    # construct DNN layers
    #dnn_final_input = merge_helper('concat', dnn_inputs)
    dnn_final_input = splice(*dnn_inputs)
    if dnn_final_input is not None:
        dnn_output_dim = dnn_hid_size_list[-1]
        dnn_hid = dnn_final_input
        dnn_skip_connections = []
        if dropout > 1e-7:
            dnn_hid = Dropout(dropout)(dnn_hid)
        prev_output_dim = dnn_input_total_dim
        if skip_connection:
            tmp_out = dnn_hid
            if prev_output_dim != dnn_output_dim:
                tmp_out = Dense(dnn_output_dim #,
                                        #W_regularizer=create_regularizer(),
                                        #b_regularizer=create_regularizer(),
                                       )(tmp_out)
            dnn_skip_connections.append(tmp_out)
        for (depth, hid_size) in enumerate(dnn_hid_size_list):
            layer_input = dnn_hid
            dnn_hid = Dense(hid_size #, 
                            #W_regularizer=create_regularizer(),
                            #b_regularizer=create_regularizer(),
                           )(dnn_hid)
            if batch_normalization:
                dnn_hid = BatchNormalization()(dnn_hid)
            #dnn_hid = Activation('tanh')(dnn_hid)
            dnn_hid = tanh(dnn_hid)
            if dropout > 1e-7:
                dnn_hid = Dropout(dropout)(dnn_hid)
            if residual_dnn:
                if prev_output_dim != hid_size:
                    layer_input = Dense(hid_size #,
                                        #W_regularizer=create_regularizer(),
                                        #b_regularizer=create_regularizer(),
                                       )(layer_input)
                #dnn_hid = Merge(mode='sum')([dnn_hid, layer_input])
                dnn_hid = dnn_hid + layer_input
            prev_output_dim = hid_size
            if skip_connection:
                tmp_out = dnn_hid
                if prev_output_dim != dnn_output_dim:
                    tmp_out = Dense(dnn_output_dim #,
                                            #W_regularizer=create_regularizer(),
                                            #b_regularizer=create_regularizer(),
                                           )(tmp_out)
                dnn_skip_connections.append(tmp_out)
        if skip_connection:
            #dnn_hid = merge_helper('sum', dnn_skip_connections)
            dnn_hid = plus(*dnn_skip_connections)
        else:
            pass
    else:
        dnn_hid = None
    
    # merge RNN with DNN and project to final
    #rnn_dnn_merged = merge_helper('concat', rnn_hid_list + [dnn_hid])
    rnn_dnn_merged = splice(*rnn_hid_list, dnn_hid)
    #assert rnn_dnn_merged is not None, "Error! no inputs found!"
    #output = Dense(1, W_regularizer=create_regularizer(),
    #                  b_regularizer=create_regularizer(),)(rnn_dnn_merged)
    output = Dense(1)(rnn_dnn_merged)
    if batch_normalization:
        output = BatchNormalization()(output)
    #output = Activation('sigmoid')(output)
    output = sigmoid(output)
    
    # final specify model
    #model = Model(input=rnn_inputs + dnn_inputs, output=output)
    model = output  # Model(input=rnn_inputs + dnn_inputs, output=output)
    debughelpers.dump_function(model, "model")
    #logger.info(model.get_config())
    logger.info("build model completed")
    if model_save_path:
        #plot(model, to_file=model_save_path + '.png')
        from cntk.graph import output_function_graph
        output_function_graph(model, pdf_file_path=model_save_path + '.pdf', scale=2)
        #output_function_graph(model, svg_file_path=model_save_path + '.svg', scale=2)
        logger.info('model graph saved to ' + model_save_path + '.png')        
    return model

### END UNRELATED TEST



if __name__ == '__main__':

    from _cntk_py import set_computation_network_trace_level, set_fixed_random_seed, force_deterministic_algorithms
    set_fixed_random_seed(1)  # BUGBUG: has no effect at present  # TODO: remove debugging facilities once this all works

    #build_model_xy('c:/me/xinying_graph')

    #x = placeholder_variable('x')   # Function argument in definition
    #s = x * x                       # Function
    #debughelpers.dump_function(s)
    #arg = placeholder_variable('arg') # apply Function to another placeholder
    #y = s.clone(CloneMethod.share, {x: arg})
    #debughelpers.dump_function(y)
    #print(13)


    #x = placeholder_variable('x') # Function argument
    #h_f = placeholder_variable('h_f') # recurrent forward reference
    #h = sigmoid(2 * h_f + 2 * x)
    #h.replace_placeholders({h_f: h}) # end of Function definition
    #h.replace_placeholders({x: input_variable(300)}) # Function application
    #debughelpers.dump_function(h)
    #print(13)

    # test for multi-input plus()
    #from cntk.ops import plus, element_times, max, min, log_add_exp
    #for op in (log_add_exp, max, min, plus, element_times):
    #    s4 = op(Placeholder(name='a'), Placeholder(3, name='b'), Placeholder(4, name='c'), Placeholder(5, name='d'), name='s4')
    #    s4.dump('s4')
    #sequence_reduce_max = Fold(max)
    #sequence_reduce_max.dump('sequence_reduce_max')
    # TODO: create proper test case for this


    #L = Dense(500)
    #L1 = L.clone(CloneMethod.clone)
    #x = placeholder_variable()
    #y = L(x) + L1(x)

    #L = Dense(500)
    #o = L.outputs
    #sh = L.shape
    #W = L.W
    #w = L.weights

    # repro for as_block
    from cntk import placeholder_variable, combine, alias, as_block
    def f(x,y):
        return y-x
    arg_names = ['x', 'y']
    args = [placeholder_variable(name=name) for name in arg_names]
    block_args = [placeholder_variable(name=arg.name) for arg in args]  # placeholders inside the BlockFunction
    combined_block_args = combine(block_args)                           # the content of the BlockFunction
    arg_map = list(zip(block_args, args))                               # after wrapping, the block_args map to args
    combined_args = as_block(composite=combined_block_args, block_arguments_map=arg_map, block_op_name='f_parameter_pack')
    funargs = combined_args.outputs       # the Python function is called with these instead
    #combined_args=None
    out = f(*funargs)
    out_arg_names = [arg.name for arg in out.arguments]
    #out = Recurrence(out, initial_state=13.0)
    #out_arg_names = [arg.name for arg in out.arguments]
    out = out.clone(CloneMethod.share, {out.arguments[0]: input_variable(1, name='x1'), out.arguments[1]: input_variable(1, name='y1')})
    out_arg_names = [arg.name for arg in out.arguments]
    res = out.eval({out.arguments[0]: [[3.0]], out.arguments[1]: [[5.0]]})
    #res = out.eval([[3.0]])

    # repro for name loss
    #from cntk import plus, as_block
    #from _cntk_py import InferredDimension
    #arg = placeholder_variable()
    #x = times(arg, parameter((InferredDimension,3), init=glorot_uniform()), name='x')
    #x = sequence.first(x)
    #sqr = x*x
    #x1 = sqr.find_by_name('x')
    #sqr2 = as_block(sqr, [(sqr.placeholders[0], placeholder_variable())], 'sqr')
    #sqr2 = combine([sqr2])
    #x2 = sqr2.find_by_name('x')

    #stest = sqr2
    ##stest.dump()
    #stest = stest.replace_placeholders({stest.arguments[0]: input_variable(13)})
    ##stest.dump()

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
