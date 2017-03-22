# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import numpy as np
import os
from cntk import Trainer, Axis
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT, FULL_DATA_SWEEP
from cntk.learners import momentum_sgd, adam_sgd, momentum_as_time_constant_schedule, learning_rate_schedule, UnitType
from cntk import input, cross_entropy_with_softmax, classification_error, sequence, past_value, future_value, \
                 element_select, alias, hardmax, placeholder, combine, parameter, times, plus
from cntk.ops.functions import CloneMethod, load_model, Function
from cntk.initializer import glorot_uniform
from cntk.logging import log_number_of_parameters, ProgressPrinter
from cntk.logging.graph import plot
from cntk.layers import *
from cntk.layers.sequence import *
from cntk.layers.models.attention import *
from cntk.layers.typing import *

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
hidden_dim = 512
num_layers = 2
attention_dim = 128
attention_span = 20
attention_axis = -3
use_attention = True
use_embedding = True
embedding_dim = 200
vocab = ([w.strip() for w in open(os.path.join(DATA_DIR, VOCAB_FILE)).readlines()]) # all lines of VOCAB_FILE in a list
length_increase = 1.5

# sentence-start symbol as a constant
sentence_start = Constant(np.array([w=='<s>' for w in vocab], dtype=np.float32))
sentence_end_index = vocab.index('</s>')
# TODO: move these where they belong

model_path_stem = os.path.join(MODEL_DIR, "model_att_{}".format(use_attention)) # encode as many config vars as desired
def model_path(epoch):
    return model_path_stem + ".cmf." + str(epoch)


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

# type annotations for the two sequence types; later use InputSequence[Tensor[input_vocab_dim]]
# CNTK considers these two different types since they run over different sequence indices.
inputAxis = Axis('inputAxis')
labelAxis = Axis('labelAxis')
InputSequence = SequenceOver[inputAxis]
LabelSequence = SequenceOver[labelAxis]

# create the s2s model
def create_model(): # :: (history*, input*) -> logP(w)*
    # Embedding: (input*) --> embedded_input*
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

    # Decoder: (history*, input*) --> unnormalized_word_logp*
    # where history is one of these, delayed by 1 step and <s> prepended:
    #  - training: labels
    #  - testing:  its own output hardmax(z) (greedy decoder)
    with default_options(enable_self_stabilization=True):
        # sub-layers
        stab_in = Stabilizer()
        rec_blocks = [LSTM(hidden_dim) for i in range(num_layers)]
        stab_out = Stabilizer()
        proj_out = Dense(label_vocab_dim, name='out_proj')
        # attention model
        if use_attention: # maps a decoder hidden state and the entire encoder state into an augmented decoder state
            attention_model = AttentionModel(attention_dim, attention_span, attention_axis, name='attention_model') # :: (h_enc*, h_dec) -> (h_dec augmented)
        # layer function
        @Function
        def decode(history, input):
            encoded_input = encode(input)
            r = history
            r = embed(r)
            r = stab_in(r)
            for i in range(num_layers):
                rec_block = rec_blocks[i]   # LSTM(hidden_dim)  # :: (dh, dc, x) -> (h, c)
                if use_attention:
                    if i == 0:
                        @Function
                        def lstm_with_attention(dh, dc, x):
                            h_att = attention_model(encoded_input.outputs[0], dh)
                            x = splice(x, h_att) # TODO: should this be added instead? (cf. BS example)
                            return rec_block(dh, dc, x)
                        r = Recurrence(lstm_with_attention)(r)
                    else:
                        r = Recurrence(rec_block)(r)
                else:
                    # unlike Recurrence(), the RecurrenceFrom() layer takes the initial hidden state as a data input
                    r = RecurrenceFrom(rec_block)(*(encoded_input.outputs + (r,))) # :: h0, c0, r -> h  (Python < 3.5)
                    #r = RecurrenceFrom(rec_block)(*encoded_input.outputs, r) # :: h0, c0, r -> h  (Python 3.5+)
            r = stab_out(r)
            r = proj_out(r)
            r = Label('out_proj_out')(r)
            return r

    return decode

########################
# train action         #
########################

def create_model_train(s2smodel):
    # model used in training (history is known from labels)
    # note: the labels must not contain the initial <s>
    @Function
    def model_train(input, labels): # (input*, labels*) --> (word_logp*)

        # The input to the decoder always starts with the special label sequence start token.
        # Then, use the previous value of the label sequence (for training) or the output (for execution).
        # BUGBUG: This will currently fail with sparse input.
        past_labels = Delay(initial_state=sentence_start)(labels)
        return s2smodel(past_labels, input)
    return model_train

def create_model_greedy(s2smodel):
    # model used in (greedy) decoding (history is decoder's own output)
    @Function
    @Signature(InputSequence[Tensor[input_vocab_dim]])
    def model_greedy(input): # (input*) --> (word_sequence*)

        # Decoding is an unfold() operation starting from sentence_start.
        # We must transform s2smodel (history*, input* -> word_logp*) into a generator (history* -> output*)
        # which holds 'input' in its closure.
        unfold = UnfoldFrom(lambda history: s2smodel(history, input) >> hardmax,
                            until_predicate=lambda w: w[...,sentence_end_index],  # stop once sentence_end_index was max-scoring output
                            length_increase=length_increase, initial_state=sentence_start)
        # TODO: The signature should be changed, so that the initial_state is passed as data.
        return unfold(dynamic_axes_like=input)
    return model_greedy

def create_criterion_function(model):
    @Function
    @Signature(input = InputSequence[Tensor[input_vocab_dim]], labels = LabelSequence[Tensor[label_vocab_dim]])
    def criterion(input, labels):
        # criterion function must drop the <s> from the labels
        postprocessed_labels = sequence.slice(labels, 1, 0) # <s> A B C </s> --> A B C </s>
        z = model(input, postprocessed_labels)
        ce   = cross_entropy_with_softmax(z, postprocessed_labels)
        errs = classification_error      (z, postprocessed_labels)
        return (ce, errs)

    # use the following to render the Function graph to a PDF file
    #plot(criterion, filename=os.path.join(MODEL_DIR, "model") + '.pdf')
    return criterion

# dummy for printing the input sequence below. Currently needed because input is sparse.
def create_sparse_to_dense(input_vocab_dim):
    I = Constant(np.eye(input_vocab_dim))
    @Function
    @Signature(InputSequence[SparseTensor[input_vocab_dim]])
    def no_op(input):
        return times(input, I)
    return no_op

def train(train_reader, valid_reader, vocab, i2w, s2smodel, max_epochs, epoch_size):

    # Note: We would like to set the signature of 's2smodel' (s2smodel.update_signature()), but that will cause
    # an error since the training criterion uses a reduced sequence axis for the labels.
    # This is because it removes the initial <s> symbol. Hence, we must leave the model
    # with unspecified input shapes and axes.

    # create the training wrapper for the s2smodel, as well as the criterion function
    model_train = create_model_train(s2smodel)
    criterion = create_criterion_function(model_train)

    # also wire in a greedy decoder so that we can properly log progress on a validation example
    # This is not used for the actual training process.
    model_greedy = create_model_greedy(s2smodel)

    # This does not need to be done in training generally though
    # Instantiate the trainer object to drive the model training
    minibatch_size = 72
    lr = 0.001 if use_attention else 0.005   # TODO: can we use the same value for both?
    learner = adam_sgd(model_train.parameters,
                       lr       = learning_rate_schedule([lr]*2+[lr/2]*3+[lr/4], UnitType.sample, epoch_size),
                       momentum = momentum_as_time_constant_schedule(1100),
                       gradient_clipping_threshold_per_sample=2.3,
                       gradient_clipping_with_truncation=True)
    trainer = Trainer(None, criterion, learner)

    # Get minibatches of sequences to train with and perform model training
    total_samples = 0
    mbs = 0
    eval_freq = 100

    # print out some useful training information
    log_number_of_parameters(model_train) ; print()
    progress_printer = ProgressPrinter(freq=30, tag='Training')
    #progress_printer = ProgressPrinter(freq=30, tag='Training', log_to_file=model_path_stem + ".log") # use this to log to file

    sparse_to_dense = create_sparse_to_dense(input_vocab_dim)

    for epoch in range(max_epochs):
        print("Saving model to '%s'" % model_path(epoch))
        s2smodel.save(model_path(epoch))

        while total_samples < (epoch+1) * epoch_size:
            # get next minibatch of training data
            mb_train = train_reader.next_minibatch(minibatch_size)
            #trainer.train_minibatch(mb_train[train_reader.streams.features], mb_train[train_reader.streams.labels])
            trainer.train_minibatch({criterion.arguments[0]: mb_train[train_reader.streams.features], criterion.arguments[1]: mb_train[train_reader.streams.labels]})

            progress_printer.update_with_trainer(trainer, with_metric=True) # log progress

            # every N MBs evaluate on a test sequence to visually show how we're doing
            if mbs % eval_freq == 0:
                mb_valid = valid_reader.next_minibatch(1)

                # run an eval on the decoder output model (i.e. don't use the groundtruth)
                e = model_greedy(mb_valid[valid_reader.streams.features])
                print(format_sequences(sparse_to_dense(mb_valid[valid_reader.streams.features]), i2w))
                print("->")
                print(format_sequences(e, i2w))

                # debugging attention
                if use_attention:
                    debug_attention(model_greedy, mb_valid[valid_reader.streams.features])

            total_samples += mb_train[train_reader.streams.labels].num_samples
            mbs += 1

        # log a summary of the stats for the epoch
        progress_printer.epoch_summary(with_metric=True)

    # done: save the final model
    print("Saving final model to '%s'" % model_path(max_epochs))
    s2smodel.save(model_path(max_epochs))
    print("%d epochs complete." % max_epochs)

########################
# test decoding        #
########################

# This decodes the test set and counts the string error rate.
def evaluate_decoding(reader, s2smodel, i2w):
    
    model_decoding = create_model_greedy(s2smodel) # wrap the greedy decoder around the model

    progress_printer = ProgressPrinter(tag='Evaluation')

    sparse_to_dense = create_sparse_to_dense(input_vocab_dim)

    minibatch_size = 1024
    num_total = 0
    num_wrong = 0
    while True:
        mb = reader.next_minibatch(minibatch_size)
        if not mb: # finish when end of test set reached
            break
        e = model_decoding(mb[reader.streams.features])
        outputs = format_sequences(e, i2w)
        labels  = format_sequences(sparse_to_dense(mb[reader.streams.labels]), i2w)
        # prepend sentence start for comparison
        outputs = ["<s> " + output for output in outputs]

        num_total += len(outputs)
        num_wrong += sum([label != output for output, label in zip(outputs, labels)])
        
    rate = num_wrong / num_total
    print("string error rate of {:.1f}% in {} samples".format(100 * rate, num_total))
    return rate

#######################
# test metric         #
#######################

# helper function to create a dummy Trainer that one can call test_minibatch() on
# TODO: replace by a proper such class once available
def Evaluator(model, criterion):
    from cntk import Trainer
    from cntk.learners import momentum_sgd, learning_rate_schedule, UnitType, momentum_as_time_constant_schedule
    loss, metric = Trainer._get_loss_metric(criterion)
    parameters = set(loss.parameters)
    if model:
        parameters |= set(model.parameters)
    if metric:
        parameters |= set(metric.parameters)
    dummy_learner = momentum_sgd(tuple(parameters), 
                                 lr = learning_rate_schedule(1, UnitType.minibatch),
                                 momentum = momentum_as_time_constant_schedule(0))
    return Trainer(model, (loss, metric), dummy_learner)

# This computes the metric on the test set.
# Note that this is not decoding; just predicting words using ground-truth history, like in training.
def evaluate_metric(reader, s2smodel, num_minibatches=None):

    model_train = create_model_train(s2smodel)
    criterion = create_criterion_function(model_train)

    evaluator = Evaluator(None, criterion)

    # Get minibatches of sequences to test and perform testing
    minibatch_size = 1024
    total_samples = 0
    total_error = 0.0
    while True:
        mb = reader.next_minibatch(minibatch_size)
        if not mb: # finish when end of test set reached
            break
        #mb_error = evaluator.test_minibatch(mb[reader.streams.features], mb[reader.streams.labels])
        mb_error = evaluator.test_minibatch({criterion.arguments[0]: mb[reader.streams.features], criterion.arguments[1]: mb[reader.streams.labels]})
        num_samples = mb[reader.streams.labels].num_samples
        total_error += mb_error * num_samples
        total_samples += num_samples

        if num_minibatches != None:
            num_minibatches -= 1
            if num_minibatches == 0:
                break

    # and return the test error
    rate = total_error/total_samples
    print("error rate of {:.1f}% in {} samples".format(100 * rate, total_samples))
    return rate

########################
# interactive session  #
########################

def translate(tokens, model_decoding, vocab, i2w, show_attention=False, max_label_length=20):

    vdict = {v:i for i,v in enumerate(vocab)}
    try:
        w = [vdict["<s>"]] + [vdict[c] for c in tokens] + [vdict["</s>"]]
    except:
        print('Input contains an unexpected token.')
        return []

    # convert to one_hot
    query = Value.one_hot([w], len(vdict))
    pred = model_decoding(query)
    pred = pred[0] # first sequence (we only have one) -> [len, vocab size]
    if use_attention:
        pred = pred[:,0,0,:] # attention has extra dimensions

    # print out translation and stop at the sequence-end tag
    prediction = np.argmax(pred, axis=-1)
    translation = [i2w[i] for i in prediction]
    
    # show attention window (requires matplotlib, seaborn, and pandas)
    if use_attention and show_attention:
    
        #att_value = model_decoding.attention_model.attention_weights(query)
        # BUGBUG: fails with "Forward: Feature Not Implemented"
        q = combine([model_decoding.attention_model.attention_weights])
        att_value = q(query)

        # get the attention data up to the length of the output (subset of the full window)
        att_value = att_value[0,0:len(prediction),0:len(w),0,0] # -> (len, span)

        # set up the actual words/letters for the heatmap axis labels
        columns = [i2w[ww] for ww in prediction]
        index = [i2w[ww] for ww in w]

        # show the attention weight heatmap
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        dframe = pd.DataFrame(data=np.fliplr(att_value.T), columns=columns, index=index)
        sns.heatmap(dframe)
        print('close the heatmap window to continue')
        plt.show()

    return translation

def interactive_session(s2smodel, vocab, i2w, show_attention=False):

    model_decoding = create_model_greedy(s2smodel) # wrap the greedy decoder around the model

    import sys

    print('Enter one or more words to see their phonetic transcription.')
    while True:
        line = input("> ")
        if line.lower() == "quit":
            break
        # tokenize. Our task is letter to sound.
        out_line = []
        for word in line.split():
            in_tokens = [c.upper() for c in word]
            out_tokens = translate(in_tokens, model_decoding, vocab, i2w, show_attention=True)
            out_line.extend(out_tokens)
        out_line = [" " if tok == '</s>' else tok[1:] for tok in out_line]
        print("=", " ".join(out_line))
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
def format_sequences(sequences, i2w):
    return [" ".join([i2w[np.argmax(w)] for w in s]) for s in sequences]

# to help debug the attention window
def debug_attention(model, input):
    q = combine([model, model.attention_model.attention_weights])
    #words, p = q(input) # Python 3
    words_p = q(input)
    words = words_p[0]
    p     = words_p[1]
    len = words.shape[attention_axis-1]
    span = 7 #attention_span  #7 # test sentence is 7 tokens long
    p_sq = np.squeeze(p[0,:len,:span,0,:]) # (batch, len, attention_span, 1, vector_dim)
    opts = np.get_printoptions()
    np.set_printoptions(precision=5)
    print(p_sq)
    np.set_printoptions(**opts)

#############################
# main function boilerplate #
#############################

if __name__ == '__main__':
    #try_set_default_device(cpu())

    from _cntk_py import set_fixed_random_seed
    set_fixed_random_seed(1)

    # hook up data
    vocab, i2w, w2i = get_vocab(os.path.join(DATA_DIR, VOCAB_FILE))

    # create inputs and create model
    model = create_model()
    
    # train
    train_reader = create_reader(os.path.join(DATA_DIR, TRAINING_DATA), True)
    valid_reader = create_reader(os.path.join(DATA_DIR, VALIDATION_DATA), True)
    train(train_reader, valid_reader, vocab, i2w, model, max_epochs=30, epoch_size=908241)

    test_epoch = 10
    model = Function.load(model_path(test_epoch))

    # test string error rate on decoded output
    test_reader = create_reader(os.path.join(DATA_DIR, TESTING_DATA), False)
    evaluate_decoding(test_reader, model, i2w)
    
    # test same metric same as in training on test set
    test_reader = create_reader(os.path.join(DATA_DIR, TESTING_DATA), False)
    evaluate_metric(test_reader, model)

    # try the model out in an interactive session
    interactive_session(model, vocab, i2w, show_attention=True)
