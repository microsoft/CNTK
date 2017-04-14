# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import sys
import os
from cntk import Trainer, Axis #, text_format_minibatch_source, StreamConfiguration
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT
from cntk.device import cpu, try_set_default_device
from cntk.learners import sgd, learning_rate_schedule, UnitType
from cntk.ops import input, sequence, relu
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.logging import *
from cntk import debugging
from cntk.layers import *
import cntk
import dynamite
import typing

input_dim = 2000
hidden_dim = 25
embedding_dim = 50
num_output_classes = 5

# Create the reader
def create_reader(path, is_training, input_dim, label_dim):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        features = StreamDef(field='x', shape=input_dim,   is_sparse=True),
        labels   = StreamDef(field='y', shape=label_dim,   is_sparse=False)
    )), randomize=is_training, max_sweeps = INFINITELY_REPEAT if is_training else 1)

cached_eyes = dict()

# convert CNTK reader's minibatch to our internal representation
def from_cntk_mb(inputs: tuple, variables: tuple):
    def convert(self, var): # var is for reference to know the axis
        data = self.data
        # unpack MBLayout
        sequences, _ = data.unpack_variable_value(var, True, data.device)
        # turn into correct NDArrayView types
        has_axis = len(var.dynamic_axes) > 1
        def fix_up(data):
            data.__class__ = cntk.core.NDArrayView # data came in as base type
            shape = data.shape
            # map to dense if sparse for now, since we cannot batch sparse due to lack of CUDA kernel
            if data.is_sparse:
                global cached_eyes
                dim = shape[1] # (BUGBUG: won't work for >1D sparse objects)
                if dim not in cached_eyes:
                    eye_np = np.array(np.eye(dim), np.float32)
                    cached_eyes[dim] = cntk.NDArrayView.from_dense(eye_np)
                eye = cached_eyes[dim]
                data = data @ eye
                assert shape == data.shape
            else: # if dense then we clone it so that we won't hold someone else's reference
                data = data.deep_clone()
                data.__class__ = cntk.core.NDArrayView
            item_shape = shape[1:]  # drop a superfluous length dimension
            if has_axis:
                seq = dynamite.Constant(data) # turn into a tensor object
                res = [seq[t] for t in range(shape[0])] # slice it
                return res
            else:
                assert shape[0] == 1
                return dynamite.Constant(data[0])
        return [fix_up(seq) for seq in sequences]
    return tuple(convert(inp, var) for inp, var in zip(inputs, variables))

# Define the LSTM model for classifying sequences
cntk.Sequential = cntk.layers.Sequential  # all in one namespace, to use same code for CNTK and dynamite
cntk.Embedding = cntk.layers.Embedding
cntk.Fold = cntk.layers.Fold
cntk.RNNUnit = cntk.layers.RNNUnit
cntk.Dense = cntk.layers.Dense
cntk.identity = cntk.layers.identity
cntk.LogValues = lambda: cntk.layers.identity
def create_model(namespace, num_output_classes, embedding_dim, hidden_dim):
    return namespace.Sequential([
        namespace.Embedding(embedding_dim, name='embed'),
        namespace.Fold(namespace.RNNUnit(hidden_dim, activation=namespace.relu, name='rnn')),
        #namespace.identity,
        namespace.LogValues(),
        namespace.Dense(num_output_classes, name='dense')
    ])

# define the criterion fnction
# note: not using @Function here since using the same for dynamite
def create_criterion(namespace, model):
    def criterion(input: Sequence[SparseTensor[input_dim]], label: Tensor[num_output_classes]):
        z = model(input)
        ce = namespace.cross_entropy_with_softmax(z, label)
        pe = namespace.classification_error(z, label)
        return (ce, pe)
    return criterion

# Create and train a LSTM sequence classification model
def train(debug_output=False):
    # Input variables denoting the features and label data
    #features = sequence.input(shape=input_dim, is_sparse=True)
    #label = input(num_output_classes)

    # Instantiate the sequence classification model
    model = create_model(cntk, num_output_classes, embedding_dim, hidden_dim)
    dmodel = create_model(dynamite, num_output_classes, embedding_dim, hidden_dim)

    criterion = Function(create_criterion(cntk, model))
    dcriterion = create_criterion(dynamite, dmodel)
    debugging.dump_signature(criterion)

    # share static model's parameters over to dynamic model
    # Note: This must be done in exactly this order for the matrices, otherwise it affects the result. Seems some random init happens here.
    dmodel.__items__[3].W              .share_data_from(model.dense.W)
    dmodel.__items__[3].b              .share_data_from(model.dense.b)
    dmodel.__items__[1].step_function.b.share_data_from(model.rnn.b  )
    dmodel.__items__[1].step_function.W.share_data_from(model.rnn.W  )
    dmodel.__items__[1].step_function.R.share_data_from(model.rnn.H  )
    dmodel.__items__[0].E              .share_data_from(model.embed.E)
    parameter_map = { # [static Parameter] -> dynamite.Parameter
        model.dense.W: dmodel.__items__[3].W              ,
        model.dense.b: dmodel.__items__[3].b              ,
        model.rnn.b  : dmodel.__items__[1].step_function.b,
        model.rnn.W  : dmodel.__items__[1].step_function.W,
        model.rnn.H  : dmodel.__items__[1].step_function.R,
        model.embed.E: dmodel.__items__[0].E              
    }
    dparameters = dmodel.get_parameters()
    dparam_names = dmodel.get_parameter_names() # [dynamite.Parameter] -> name
    #for p, dp in parameter_map.items():
    #    print(p.shape, dp.shape, p.name, dparam_names[dp])
    print('dynamic model has', len(dparameters), 'parameter tensors:', ', '.join(name + str(param.shape) for param, name in dparam_names.items()))

    # testing stuff
    if False:
        m1 = dynamite.Dense(2, activation=dynamite.relu)
        dp1 = m1.get_parameters()
        x = dynamite.Constant(np.array([1., 2., 3.]))
        l = dynamite.Constant(np.array([1., 0.]))
        s = dynamite.cross_entropy_with_softmax(m1(x), l)
        r = s.to_ndarray()
        #dynamite.dump_graph(s)
        g = s.grad_times(dp1)
        for p in dp1:
            gp = g[p]
            #dynamite.dump_graph(gp)
            print(gp.to_ndarray())

    rel_path = "../CNTK/Tests/EndToEndTests/Text/SequenceClassification/Data/Train.ctf"
    reader = create_reader(os.path.dirname(os.path.abspath(__file__)) + '/' + rel_path, True, input_dim, num_output_classes)

    lr_per_sample = learning_rate_schedule(0.05, UnitType.sample)
    # Instantiate the trainer object to drive the model training
    learner = sgd(model.parameters, lr=lr_per_sample)
    trainer = Trainer(None, criterion, learner)

    # process minibatches and perform model training
    training_progress_output_freq = 10
    log_number_of_parameters(model) ; print()
    progress_printer = ProgressPrinter(freq=training_progress_output_freq, first=10, tag='Training') # more detailed logging
    #progress_printer = ProgressPrinter(tag='Training')

    # Get minibatches of sequences to train with and perform model training
    minibatch_size = 200

    if debug_output:
        training_progress_output_freq = training_progress_output_freq/3

    expected_losses = [ 1.589759, 1.497586, 1.520561, 2.354308, 1.712215, 1.268698, 1.161497, 1.473921, 1.283860, 1.153169 ] # first 10 MBs from CNTK static

    import time
    for i in range(251):
        mb = reader.next_minibatch(minibatch_size)

        def log_time(dur):
            dur_per_sample = dur / len(args[0])
            samples_per_second = 1 / dur_per_sample
            #print('{:.2f} ms, {:.1f} samples/s'.format(dur * 1000, samples_per_second))

        # CNTK dynamite  --do this first before CNTK updates anything
        args = from_cntk_mb((mb[reader.streams.features], mb[reader.streams.labels]), criterion.arguments)
        gstart = time.time()
        crit = dynamite.train_minibatch(dcriterion, *args)
        gend = time.time()
        log_time(gend-gstart)
        dstart = time.time()
        crit_nd = crit.to_ndarray()
        dend = time.time()
        log_time(dend-dstart)
        loss = crit_nd / len(args[0])
        print(" " * 29, loss)
        if expected_losses: # test
            loss_ex, *expected_losses = expected_losses
            assert np.allclose(loss, loss_ex, atol=1e-5)
            print('ok')

        #dynamite.dump_graph(crit, skip_free=True)
        #exit()
        # compute gradients
        dgradients = crit.grad_times(dparameters)
        #dynamite.batch_eval([dgradients[p] for p in dparameters]) # compute all in a single shot, to see if it makes a difference --does not
        #for p in dparameters:
        #    print('gradient for', dparam_names[p])
        #    g = dgradients[p].get_value()
            #print(g.to_ndarray())

        if True:
            # CNTK static, manual fw/bw/update
            grads = combine([criterion.outputs[0]]).grad(at=criterion.argument_map(mb[reader.streams.features], mb[reader.streams.labels]), wrt=model.parameters, as_numpy=False)
            for p in model.parameters:
                dp = parameter_map[p] # map parameter from static to Dynamite gradients
                dpname = dparam_names[dp]
                print(dpname)
                if dpname == '_[0].E':
                    continue  # cannot convert sparse gradient to numpy
                dp = dgradients[dp] # find the gradient for the parameter
                #print('### gradient for', dpname, '(CNTK static vs. dynamite)')
                p_data = grads[p].data.to_ndarray()
                #dynamite.VariableGlobalConfig.enable_tracing = True
                dp_data = dp.to_ndarray() # this will trigger computation
                #print(p_data)
                #print(dp_data)
                #exit()
                # Dense.W fails when not using batching; but is OK without batching, so some gradient is just wrong
                assert np.allclose(p_data, dp_data, atol=1e-5)
                if dpname == "_[1].step_function.W":
                    dynamite.dump_graph(dp, skip_free=True)
                    exit()

            # model update from dynamic
            param_map = { p: dgradients[parameter_map[p]].get_value() for p in model.parameters }
            #for p, g in param_map.items():
            #    print(p.shape, g.shape, p.name)
            learner.update(param_map, len(args[0]))
        else:
            # CNTK static, original example
            start = time.time()
            trainer.train_minibatch(criterion.argument_map(mb[reader.streams.features], mb[reader.streams.labels]))
            progress_printer.update_with_trainer(trainer, with_metric=True)    # log progress
            end = time.time()
            log_time(end-start)
    loss, metric, actual_samples = progress_printer.epoch_summary(with_metric=True)

    import copy

    evaluation_average = copy.copy(
        trainer.previous_minibatch_evaluation_average)
    loss_average = copy.copy(trainer.previous_minibatch_loss_average)

    return evaluation_average, loss_average

if __name__ == '__main__':
    # Specify the target device to be used for computing, if you do not want to
    # use the best available one, e.g.
    # try_set_default_device(cpu())

    train()
