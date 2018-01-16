# =============================================================================
# copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
import ctypes
import numpy as np
import cntk as C
import argparse
import os
import datetime
import platform

from cntk.train.distributed import Communicator
from cntk.train.distributed import data_parallel_distributed_learner

from cntk.layers import sequence, Stabilizer, Embedding, Dense, Dropout, Sequential, For
from cntk.logging import log_number_of_parameters

from lightrnn import lightlstm
from math import ceil, sqrt
from data_reader import DataSource
from functools import reduce
from operator import add
from ctypes import c_double, create_string_buffer
from reallocate import reallocate_table


parser = argparse.ArgumentParser(description="Language Model with LightRNN")

# The folder
parser.add_argument('-datadir', '--datadir', default=None, required=True,
                    help='Data directory where the dataset is located')
parser.add_argument('-outputdir', '--outputdir', default='Models/',
                    help='Output directory for checkpoints and models')
parser.add_argument('-vocabdir', '--vocabdir', default='WordInfo',
                    help='Vocab directory where put the word location')

# The file
parser.add_argument('-vocab_file', '--vocab_file', default=None, required=True,
                    help='The path of vocabulary file')
parser.add_argument('-train_file', '--train_file', default='train.txt',
                    help='The name of train file')
parser.add_argument('-valid_file', '--valid_file', default='valid.txt',
                    help='The name of valid file')
parser.add_argument('-test_file', '--test_file', default='test.txt',
                    help='The name of test file')
parser.add_argument('-alloc_file', '--alloc_file', default='word-0.location', type=str,
                    help='The path of word location')
parser.add_argument('-pre_model', '--pre_model', default=None,
                    help='The pre-trained model')

# The model training parameters
parser.add_argument('-batchsize', '--batchsize', default=20, type=int,
                    help='The minibatch size')
parser.add_argument('-embed', '--embed', default=512, type=int,
                    help='The dimension of word embedding')
parser.add_argument('-nhid', '--nhid', default=512, type=int,
                    help='The dimension of hidden layer')
parser.add_argument('-layer', '--layer', default=2, type=int,
                    help='The number of rnn layer')
parser.add_argument('-dropout', '--dropout', default=0.2, type=float,
                    help='The dropout rate')
parser.add_argument('-lr', '--lr', default=0.15, type=float,
                    help='The learning rate')
parser.add_argument('-optim', '--optim', default='adam', choices=['sgd', 'adam', 'adagrad'],
                    help='The optimize method')
parser.add_argument('-seqlength', '--seqlength', default=32, type=int,
                    help='The length of sequence')
parser.add_argument('-vocabsize', '--vocabsize', default=10000, type=int,
                    help='The size of vocabulary')

# other parameters
parser.add_argument('-epochs', '--epochs', default=0, type=int, nargs='+',
                    help='The epochs number of every round')
parser.add_argument('-freq', '--freq', default=100, type=int,
                    help='The frequency of the print progress')
parser.add_argument('-save', '--save', default='model.dnn', type=str,
                    help='The save prefix')

opt = parser.parse_args()
print(opt)

vocab_sqrt = int(ceil(sqrt(opt.vocabsize)))


def prepare_dir():
    # prepare for training directory
    if not os.path.exists(opt.vocabdir):
        os.makedirs(opt.vocabdir)
        print("created dir : %s" % (opt.vocabdir))
    if not os.path.exists(opt.outputdir):
        os.makedirs(opt.outputdir)
        print("created dir : %s" % (opt.outputdir))


def get_k_round_location_path(k):
    # Get the k-round location path
    return os.path.join(opt.vocabdir, 'word-%d.location' % (k))


###########################
# Word allocate algorithm #
###########################

# The word allocate algorithm which implement by c++ to speed up
# Params:
#   row: row loss vector
#   col: col loss vector
#   vocab_size: The vocabulary size
#   vocab_base: The sqrt of vocabsize
#   word_path: the vocab file
#   save_location_path: the new location save path
def allocate_table(row, col, vocab_size, vocab_base, word_path, save_location_path):
    if platform.system() == 'Linux':
        dll_name = 'libpyreallocate.so'
    else:
        dll_name = 'libpyreallocate.dll'
    path_dir = os.path.split(os.path.realpath(__file__))[0]
    dll_path = os.path.join(path_dir, dll_name)
    if not os.path.exists(dll_path):
        print('Use a slower implementation.')
        reallocate_table(row, col, vocab_size, vocab_base, save_location_path, word_path)
        return
    lib = ctypes.cdll.LoadLibrary(dll_path)
    row = np.concatenate(row)
    col = np.concatenate(col)
    row_size = len(row)
    row = (c_double * row_size)(*row)
    col_size = len(col)
    col = (c_double * col_size)(*col)
    word_path = create_string_buffer(word_path.encode('utf-8'))
    save_location_path = create_string_buffer(save_location_path.encode('utf-8'))
    lib.allocate_table(row, col, vocab_size, vocab_base, save_location_path, word_path)


##########################
# define the graph model #
##########################

def create_model(input_dim):
    row = sequence.input_variable(shape=input_dim)
    col = sequence.input_variable(shape=input_dim)
    rowh = Sequential([Embedding(opt.embed), Stabilizer(), Dropout(opt.dropout)])(row)
    colh = Sequential([Embedding(opt.embed), Stabilizer(), Dropout(opt.dropout)])(col)

    x = C.splice(rowh, colh, axis=-1)
    x = lightlstm(opt.embed, opt.nhid)(x)
    x = For(range(opt.layer-1), lambda: lightlstm(opt.nhid, opt.nhid))(x)
    rowh = C.slice(x, -1, opt.nhid * 0, opt.nhid * 1)
    colh = C.slice(x, -1, opt.nhid * 1, opt.nhid * 2)

    row_predict = Sequential([Dropout(opt.dropout), Dense(input_dim)])(rowh)
    col_predict = Sequential([Dropout(opt.dropout), Dense(input_dim)])(colh)

    # variable : row label and col label
    row_label = sequence.input_variable(shape=input_dim)
    col_label = sequence.input_variable(shape=input_dim)
    model = C.combine([row_predict, col_predict])

    return {'row':       row,
            'col':       col,
            'row_label': row_label,
            'col_label': col_label,
            'model':     model}


#######################
# define the criteria #
#######################
# compose model into criterion functin
# return: Function: (input1, input2, label1, label2) -> (loss, metric)
def create_criterion(network):
    '''Create the criterion for model'''
    model, label1, label2 = network['model'], network['row_label'], network['col_label']
    label1_ce = C.cross_entropy_with_softmax(model.outputs[0], label1)
    label2_ce = C.cross_entropy_with_softmax(model.outputs[1], label2)
    label1_pe = C.classification_error(model.outputs[0], label1)
    label2_pe = C.classification_error(model.outputs[1], label2)
    label_ce = label1_ce + label2_ce
    label_pe = label1_pe + label2_pe
    return (label_ce, label_pe)


###########################
# define the optim method #
###########################

# create learners by params
# return: learners: [sgd, adam, adagrad]
def create_learner(model):
    '''Create the optimized method'''
    lr_per_minibatch = C.learning_parameter_schedule(opt.lr)
    momentum_schedule = C.momentum_schedule_per_sample(0.9990913221888589)
    if opt.optim == 'sgd':
        return C.sgd(model.parameters, lr=lr_per_minibatch)
    elif opt.optim == 'adam':
        return C.adam(model.parameters, lr=lr_per_minibatch, momentum=momentum_schedule)
    elif opt.optim == 'adagrad':
        return C.adagrad(model.parameters, lr=lr_per_minibatch)
    else:
        raise RuntimeError("Invalid optim method: " + opt.optim)


###################
# Evaluate action #
###################

# return : loss of eval set
def evaluate(network, path, location_path):
    criterion = create_criterion(network)
    ce = criterion[0]
    source = DataSource(path, opt.vocab_file, location_path,
                        opt.seqlength, opt.batchsize)
    error, tokens = 0, 0
    flag = True
    while flag:
        mb = source.next_minibatch(opt.seqlength * opt.batchsize)
        loss = ce.eval({
            network['row']: mb[source.input1],
            network['col']: mb[source.input2],
            network['row_label']: mb[source.label1],
            network['col_label']: mb[source.label2]
        })
        error += sum([reduce(add, _)[0] for _ in loss])
        tokens += mb[source.input1].num_samples
        flag = not mb[source.input1].sweep_end
    return error / tokens


############################
# calcuate the loss vector #
############################

# evaluate the loss vector from train data
# return row and col probability distribution on location
def calculate_loss_vector(network, path, location_path, communicator):
    source = DataSource(path, opt.vocab_file, location_path,
                        opt.seqlength, opt.batchsize)
    # the curr row -> the curr col
    # the curr col -> the next row
    row_loss = C.log(C.softmax(network['model'].outputs[0]))
    col_loss = C.log(C.softmax(network['model'].outputs[1]))
    loss = C.combine([row_loss, col_loss])
    row_loss_vector = np.zeros((opt.vocabsize, vocab_sqrt))
    col_loss_vector = np.zeros((opt.vocabsize, vocab_sqrt))

    flag = True
    while flag:
        mb = source.next_minibatch(opt.seqlength * opt.batchsize * Communicator.num_workers(),
                                   Communicator.num_workers(),
                                   communicator.rank())
        result = loss.eval({
            network['row']: mb[source.input1],
            network['col']: mb[source.input2],
        })
        row_prob = result[loss.outputs[0]]
        col_prob = result[loss.outputs[1]]
        label1 = mb[source.word1].asarray()
        label2 = mb[source.word2].asarray()
        sequences = len(label1)
        for i in range(sequences):
            seqlength = len(row_prob[i])
            for j in range(seqlength):
                row_word = int(label1[i][j][0])
                col_word = int(label2[i][j][0])
                row_loss_vector[row_word] -= row_prob[i][j]
                col_loss_vector[col_word] -= col_prob[i][j]
        flag = not mb[source.input1].sweep_end
    return col_loss_vector, row_loss_vector


################
# Train action #
################

def train(network, location_path, id):
    train_path = os.path.join(opt.datadir, opt.train_file)
    valid_path = os.path.join(opt.datadir, opt.valid_file)
    test_path = os.path.join(opt.datadir, opt.test_file)

    criterion = create_criterion(network)
    ce, pe = criterion[0], criterion[1]
    learner = create_learner(network['model'])
    learner = data_parallel_distributed_learner(learner)
    communicator = learner.communicator()
    trainer = C.Trainer(network['model'], (ce, pe), learner)

    # loop over epoch
    for epoch in range(opt.epochs[id]):
        source = DataSource(train_path, opt.vocab_file, location_path,
                            opt.seqlength, opt.batchsize)
        loss, metric, tokens, batch_id = 0, 0, 0, 0
        start_time = datetime.datetime.now()
        flag = True

        # loop over minibatch in the epoch
        while flag:
            mb = source.next_minibatch(opt.seqlength * opt.batchsize * Communicator.num_workers(),
                                       Communicator.num_workers(),
                                       communicator.rank())
            trainer.train_minibatch({
                network['row']: mb[source.input1],
                network['col']: mb[source.input2],
                network['row_label']: mb[source.label1],
                network['col_label']: mb[source.label2]
            })
            samples = trainer.previous_minibatch_sample_count
            loss += trainer.previous_minibatch_loss_average * samples
            metric += trainer.previous_minibatch_evaluation_average * samples
            tokens += samples
            batch_id += 1
            if Communicator.num_workers() > 1:
                communicator.barrier()
            if batch_id != 0 and batch_id % opt.freq == 0:
                diff_time = (datetime.datetime.now() - start_time)
                print("Epoch {:2}: Minibatch [{:5} - {:5}], loss = {:.6f}, error = {:.6f}, speed = {:3} tokens/s".format(
                    epoch + 1, batch_id - opt.freq + 1, batch_id,
                    loss / tokens, metric / tokens, tokens // diff_time.seconds))
            flag = not mb[source.input1].sweep_end

        # Evaluation action
        if communicator.is_main():
            valid_error = evaluate(network, valid_path, location_path)
            test_error = evaluate(network, test_path, location_path)
            print("Epoch {:2} Done : Valid error = {:.6f}, Test error = {:.6f}".format(epoch + 1, valid_error, test_error))
            network['model'].save(os.path.join(opt.outputdir, 'round{}_epoch{}_'.format(id, epoch) + opt.save))
        if Communicator.num_workers() > 1:
            communicator.barrier()
    # word allocate action
    row_loss, col_loss = calculate_loss_vector(network, train_path, location_path, communicator)
    if Communicator.num_workers() > 1:
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            if communicator.is_main():
                for i in range(1, Communicator.num_workers()):
                    row_loss_i, col_loss_i = comm.recv(source=i)
                    row_loss += row_loss_i
                    col_loss += col_loss_i
            else:
                data_send = [row_loss, col_loss]
                comm.send(data_send, 0)
        except:
            raise RuntimeError("Please install mpi4py if uses multi gpus!")
        communicator.barrier()
    if communicator.is_main():
        allocate_table(row_loss, col_loss,
                       opt.vocabsize, vocab_sqrt, opt.vocab_file,
                       get_k_round_location_path(id + 1))


#################
# main function #
#################

def main():
    prepare_dir()  # create the vocab dir and model dir

    network = create_model(vocab_sqrt)
    if opt.pre_model:
        network['model'].restore(opt.pre_model)
    log_number_of_parameters(network['model'])
    location_path = os.path.join(opt.vocabdir, opt.alloc_file)
    for i in range(len(opt.epochs)):
        train(network, location_path, i)
        location_path = get_k_round_location_path(i + 1)
    Communicator.finalize()


if __name__ == "__main__":
    main()
