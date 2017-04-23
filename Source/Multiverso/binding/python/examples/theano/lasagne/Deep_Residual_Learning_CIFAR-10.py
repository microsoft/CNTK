#!/usr/bin/env python

"""
This code is adapted from Lasagne/Recipes
https://github.com/Lasagne/Recipes/blob/master/papers/deep_residual_learning/Deep_Residual_Learning_CIFAR-10.py

The MIT License (MIT)

Copyright (c) 2015 Lasagne

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Lasagne implementation of CIFAR-10 examples from "Deep Residual Learning for Image Recognition" (http://arxiv.org/abs/1512.03385)

Check the accompanying files for pretrained models. The 32-layer network (n=5), achieves a validation error of 7.42%,
while the 56-layer network (n=9) achieves error of 6.75%, which is roughly equivalent to the examples in the paper.
"""

from __future__ import print_function

import sys
import os
import time
import string
import random
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                    os.path.pardir, os.path.pardir, os.path.pardir)))

import argparse
# Parsing the arguments
parser = argparse.ArgumentParser(usage="Trains a Deep Residual Learning network\
 on cifar-10 using Lasagne.\nNetwork architecture and training parameters are as\
 in section 4.2 in 'Deep Residual Learning for Image Recognition'.")

parser.add_argument('-n', type=int, help="N: Number of stacked residual\
 building blocks per feature map (default: 5)", default=5)
parser.add_argument('-m', '--model', type=str, help="saved model file to load\
 (for validation) (default: None)", default=None)
parser.add_argument('-lr', type=float, help="initial learning rate (default:\
 0.1)", default=0.1)
parser.add_argument('-s', '--sync', type=bool, help="run multiverso in sync \
 mode (default: False)", default=False)
parser.add_argument('-b', '--batch-size', type=int, help="batch size (default:\
 False)", default=128)
parser.add_argument('-e', '--epoches', type=int, help="Number of epoches(default:\
 82)", default=82)
args = parser.parse_args()
print(args)


# MULTIVERSO: import multiverso
import multiverso as mv

# MULTIVERSO: you should call mv.init before call multiverso apis
mv.init(sync=args.sync)
# MULTIVERSO: every process has distinct worker id
worker_id = mv.worker_id()
# MULTIVERSO: mv.workers_num will return the number of workers
workers_num = mv.workers_num()
# NOTICE: To use multiple gpus, we must set the environment before import theano.
if "THEANO_FLAGS" not in os.environ:
    os.environ["THEANO_FLAGS"] = 'floatX=float32,device=gpu%d,lib.cnmem=1' % worker_id

import numpy as np
import theano
import theano.tensor as T
import lasagne
from multiverso.theano_ext.lasagne_ext import param_manager

# for the larger networks (n>=9), we need to adjust pythons recursion limit
sys.setrecursionlimit(10000)

# ##################### Load data from CIFAR-10 dataset #######################
# this code assumes the cifar dataset from 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
# has been extracted in current working directory

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_data():
    xs = []
    ys = []
    for j in range(5):
      d = unpickle('cifar-10-batches-py/data_batch_'+`j+1`)
      x = d['data']
      y = d['labels']
      xs.append(x)
      ys.append(y)

    d = unpickle('cifar-10-batches-py/test_batch')
    xs.append(d['data'])
    ys.append(d['labels'])

    x = np.concatenate(xs)/np.float32(255)
    y = np.concatenate(ys)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)

    # subtract per-pixel mean
    pixel_mean = np.mean(x[0:50000],axis=0)
    #pickle.dump(pixel_mean, open("cifar10-pixel_mean.pkl","wb"))
    x -= pixel_mean

    # create mirrored images
    X_train = x[0:50000,:,:,:]
    Y_train = y[0:50000]
    X_train_flip = X_train[:,:,:,::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train,X_train_flip),axis=0)
    Y_train = np.concatenate((Y_train,Y_train_flip),axis=0)

    X_test = x[50000:,:,:,:]
    Y_test = y[50000:]

    return dict(
        X_train=lasagne.utils.floatX(X_train),
        Y_train=Y_train.astype('int32'),
        X_test = lasagne.utils.floatX(X_test),
        Y_test = Y_test.astype('int32'),)

# ##################### Build the neural network model #######################

from lasagne.layers import Conv2DLayer as ConvLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import batch_norm

def build_cnn(input_var=None, n=5):

    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def residual_block(l, increase_dim=False, projection=False):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2,2)
            out_num_filters = input_num_filters*2
        else:
            first_stride = (1,1)
            out_num_filters = input_num_filters

        stack_1 = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        stack_2 = batch_norm(ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

        # add shortcut connections
        if increase_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None, flip_filters=False))
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]),nonlinearity=rectify)
            else:
                # identity shortcut, as option A in paper
                identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
                padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]),nonlinearity=rectify)
        else:
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]),nonlinearity=rectify)

        return block

    # Building the network
    l_in = InputLayer(shape=(None, 3, 32, 32), input_var=input_var)

    # first layer, output is 16 x 32 x 32
    l = batch_norm(ConvLayer(l_in, num_filters=16, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

    # first stack of residual blocks, output is 16 x 32 x 32
    for _ in range(n):
        l = residual_block(l)

    # second stack of residual blocks, output is 32 x 16 x 16
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)

    # third stack of residual blocks, output is 64 x 8 x 8
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)

    # average pooling
    l = GlobalPoolLayer(l)

    # fully connected layer
    network = DenseLayer(
            l, num_units=10,
            W=lasagne.init.HeNormal(),
            nonlinearity=softmax)

    return network

# ############################# Batch iterator ###############################

def iterate_minibatches(inputs, targets, batchsize, shuffle=False, augment=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if augment:
            # as in paper :
            # pad feature arrays with 4 pixels on each side
            # and do random cropping of 32x32
            padded = np.pad(inputs[excerpt],((0,0),(0,0),(4,4),(4,4)),mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0,high=8,size=(batchsize,2))
            for r in range(batchsize):
                random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32)]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]

        yield inp_exc, targets[excerpt]

# ############################## Main program ################################

def main(batch_size=128, lr=0.1, sync=False, n=5, num_epochs=82, model=None):
    # Check if cifar data exists
    if not os.path.exists("./cifar-10-batches-py"):
        print("CIFAR-10 dataset can not be found. Please download the dataset from 'https://www.cs.toronto.edu/~kriz/cifar.html'.")
        return

    # Load the dataset
    print("Loading data...")
    data = load_data()
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model
    print("Building model and compiling functions...")
    network = build_cnn(input_var, n)
    print("number of parameters in model: %d" % lasagne.layers.count_params(network, trainable=True))

    # MULTIVERSO: LasagneParamManager is a parameter manager which can
    # synchronize parameters of Lasagne with multiverso.
    lpm = param_manager.LasagneParamManager(network)

    if model is None:
        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        # add weight decay
        all_layers = lasagne.layers.get_all_layers(network)
        l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
        loss = loss + l2_penalty

        # Create update expressions for training
        # Stochastic Gradient Descent (SGD) with momentum
        params = lasagne.layers.get_all_params(network, trainable=True)
        sh_lr = theano.shared(lasagne.utils.floatX(lr))
        updates = lasagne.updates.momentum(
                loss, params, learning_rate=sh_lr, momentum=0.9)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Create a loss expression for validation/testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    if model is None:
        # launch the training loop
        print("Starting training...")
        # We iterate over epochs:
        for epoch in range(num_epochs):
            # devide the data into different process
            examples_per_worker = X_train.shape[0] / workers_num
            start_index = worker_id * (examples_per_worker)
            train_indices = np.arange(start_index, start_index + examples_per_worker)
            # shuffle training data
            np.random.shuffle(train_indices)
            rand_X_train = X_train[train_indices,:,:,:]
            rand_Y_train = Y_train[train_indices]

            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(rand_X_train, rand_Y_train, batch_size, shuffle=True, augment=True):
                train_batches += 1
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                # MULTIVERSO: when you want to commit all the delta of
                # parameters manage by LasagneParamManager and update the latest
                # parameters from parameter server, you can call this function to
                # synchronize the values
                lpm.sync_all_param()

            # And a full pass over the validation data:
            # MULTIVERSO: all the workers will synchronize at the place you call barrier
            mv.barrier()
            if mv.is_master_worker():
                val_err = 0
                val_acc = 0
                val_batches = 0
                for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False):
                    inputs, targets = batch
                    err, acc = val_fn(inputs, targets)
                    val_err += err
                    val_acc += acc
                    val_batches += 1

                # Then we print the results for this epoch:
                print("Epoch {} of {} took {:.3f}s".format(
                    epoch + 1, num_epochs, time.time() - start_time))
                print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
                print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
                print("  validation accuracy:\t\t{:.2f} %".format(
                    val_acc / val_batches * 100))

            # adjust learning rate as in paper
            # 32k and 48k iterations should be roughly equivalent to 41 and 61 epochs
            if (epoch+1) == 41 or (epoch+1) == 61:
                # TODO: because of ASGD and multiple GPU are used, so Learning
                # rate change schedule should be reconsidered
                new_lr = sh_lr.get_value() * 0.1
                print("New LR:"+str(new_lr))
                sh_lr.set_value(lasagne.utils.floatX(new_lr))

        # MULTIVERSO: all the workers will synchronize at the place you call barrier
        mv.barrier()
        if mv.is_master_worker():
            # MULTIVERSO: update the parameters before save the model
            lpm.sync_all_param()
            # dump the network weights to a file :
            np.savez('cifar10_deep_residual_model.npz', *lasagne.layers.get_all_param_values(network))
    else:
        # load network weights from model file
        with np.load(model) as f:
             param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)

    if mv.is_master_worker():
        # Calculate validation error of model:
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))

    # MULTIVERSO: You must call shutdown at the end of the file
    mv.shutdown()


def array_list(arr_list):
    for arr in arr_list:
        print(type(arr), arr.size)

    print([(type(arr), arr.size, arr.shape) for arr in arr_list])

    f = np.load("cifar10_deep_residual_model.npz")
    print([f['arr_%d' % i].size for i in range(len(f.files))])
    print([(arr.size) for arr in arr_list])
    print([(arr.dtype) for arr in arr_list])


if __name__ == '__main__':
    main(batch_size=args.batch_size, lr=args.lr, sync=args.sync, n=args.n,
        num_epochs=args.epoches, model=args.model)
