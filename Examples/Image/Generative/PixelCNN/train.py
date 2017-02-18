from __future__ import print_function
import os
import time
import math
import numpy as np

import cntk as ct
from cntk import Trainer
from cntk.learner import sgd

from pixelcnn import models as m
from pixelcnn import nn as nn
from pixelcnn import losses as l

# Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
data_path  = R"S:\src\cntk\1\cntk\Examples\Image\DataSets\CIFAR-10" #os.path.join(abs_path, "..", "..", "..", "DataSets", "CIFAR-10")
model_path = os.path.join(abs_path, "Models")

# model dimensions
image_height = 32
image_width  = 32
num_channels = 3  # RGB
num_classes  = 10

# Define the reader for both training and evaluation action.
def create_reader(map_file, is_training):
    if not os.path.exists(map_file):
        raise RuntimeError("File '%s' does not exist. Please run install_cifar10.py from DataSets/CIFAR-10 to fetch them" %
                           (map_file))

    transforms = []
    transforms += [
        ct.io.ImageDeserializer.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')
    ]
    # deserializer
    return ct.io.MinibatchSource(ct.io.ImageDeserializer(map_file, ct.io.StreamDefs(
        features = ct.io.StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
        labels   = ct.io.StreamDef(field='label', shape=num_classes))),   # and second as 'label'
        randomize=is_training)

def classification_error_256(input, label, prediction):
    return ct.relu(1.0)

def train(reader_train, reader_test, epoch_size = 50000, max_epochs = 100):

    # Input variables denoting the features and label data
    input_var    = ct.input_variable((num_channels, image_height, image_width))
    # image_target = ct.input_variable(shape=(256, 3*32*32))
    image_target = ct.input_variable(shape=(num_channels, image_height, image_width))
    label_var    = ct.input_variable((num_classes))

    # apply model to input
    input_norm = (input_var - 127.5) / 127.5 # [-1, 1]
    z = m.build_pixelcnn_pp_model(input_norm) #, per_pixel_count=100)

    # loss and metric
    ce = l.discretized_mix_logistic_loss(input_norm, z) # .softmax_256_loss(image_target, z)
    pe = classification_error_256(input_var, label_var, z)

    # training config
    minibatch_size = 8

    # Set learning parameters
    lr_per_sample    = 0.0001 
    lr_schedule      = ct.learning_rate_schedule(lr_per_sample, unit=ct.learner.UnitType.sample) #, epoch_size=epoch_size)
    mm_time_constant = 4096
    mm_schedule      = ct.learner.momentum_as_time_constant_schedule(mm_time_constant) #, epoch_size=epoch_size)
    
    # trainer object
    learner = ct.learner.adam_sgd(z.parameters, lr=lr_schedule, momentum=mm_schedule)
    #learner = ct.learner.sgd(z.parameters, lr_schedule)
    trainer = ct.Trainer(z, (ce, pe), learner)

    # define mapping from reader streams to network inputs
    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    ct.utils.log_number_of_parameters(z) ; print()
    progress_printer = ct.utils.ProgressPrinter(tag='Training') # freq=10, 

    # perform model training
    epoch_size     = 50000
    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            t0 = time.perf_counter()
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size-sample_count), input_map=input_map) # fetch minibatch.

            # One hot: 256, 3*32*32
            # image  = np.asarray(data[input_var].value, dtype=int).flatten()
            #target = np.zeros((256,) + image.shape)
            #target[image, np.arange(image.size)] = 1
            #target = np.ascontiguousarray(np.reshape(target, (-1, 1, 3, 32*32)))
            #print("Target shape: {}".format(target.shape))
            t1 = time.perf_counter()

            trainer.train_minibatch({input_var:data[input_var].value})
            t2 = time.perf_counter()

            #print("MB {} of {}: read time {} ms, train time {}.".format(sample_count / minibatch_size, epoch_size / minibatch_size, (t1-t0)*1000.0, (t2-t1)*1000.0))
            sample_count += trainer.previous_minibatch_sample_count
            progress_printer.update_with_trainer(trainer, with_metric=True)

        progress_printer.epoch_summary(with_metric=True)

if __name__=='__main__':
    reader_train = create_reader(os.path.join(data_path, 'train_map.txt'), True)
    reader_test  = create_reader(os.path.join(data_path, 'test_map.txt'), False)

    train(reader_train, reader_test)
