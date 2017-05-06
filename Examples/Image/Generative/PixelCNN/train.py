from __future__ import print_function
import os
import sys
import time
import math
import argparse
from PIL import Image

import numpy as np
import cntk as ct
import cntk.io.transforms as xforms

from pixelcnn import models as m
from pixelcnn import nn as nn
from pixelcnn import losses as l
from pixelcnn import sample as sp
from pixelcnn import plotting as plotting

# Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(abs_path, "..", "..", "DataSets", "CIFAR-10")
model_path = os.path.join(abs_path, "Models")

# model dimensions
image_height    = 32
image_width     = 32
num_channels    = 3  # RGB
num_classes     = 10
image_shape     = (num_channels, image_height, image_width)
nr_logistic_mix = 10

# Define the reader for both training and evaluation action.
def create_reader(map_file, is_training):
    if not os.path.exists(map_file):
        raise RuntimeError("File '%s' does not exist. Please run install_cifar10.py from DataSets/CIFAR-10 to fetch them" %
                           (map_file))

    transforms = []
    transforms += [
        xforms.crop(crop_type='center', crop_size=32)
    ]
    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')
    ]
    # deserializer
    return ct.io.MinibatchSource(ct.io.ImageDeserializer(map_file, ct.io.StreamDefs(
        features = ct.io.StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
        labels   = ct.io.StreamDef(field='label', shape=num_classes))),   # and second as 'label'
        randomize=is_training)

def sample(batch_size, z, inputs,  nr_mix, loss, epoch, sample_index):
    x_gen = np.zeros((batch_size,) + image_shape, dtype=np.float32)
    for y in range(image_height):
        for x in range(image_width):
            if loss == 'category':
                for c in range(num_channels):
                    new_x_gen    = z.eval({inputs:x_gen})
                    new_x_gen_np = np.asarray(sp.sample_from(new_x_gen, nr_mix, loss), dtype=np.float32)
                    x_gen[:,c,y,x] = new_x_gen_np[:,c,y,x]
            else:
                new_x_gen    = z.eval({inputs:x_gen})
                new_x_gen_np = np.asarray(sp.sample_from(new_x_gen, nr_mix, loss), dtype=np.float32)
                x_gen[:,:,y,x] = new_x_gen_np[:,:,y,x]                                

    sample_x = np.ascontiguousarray(np.transpose(x_gen, (0,2,3,1)))
    img_tile = plotting.img_tile(sample_x[:int(np.floor(np.sqrt(batch_size))**2)], aspect_ratio=1.0, border_color=1.0, stretch=True)
    img = plotting.plot_img(img_tile, title="Samples from epoch {} image {}.".format(epoch, sample_index+1))
    plotting.plt.savefig("image_{}_{}.png".format(epoch, sample_index+1))
    plotting.plt.close('all')

def train(reader_train, reader_test, model, loss, epoch_size = 50000, max_epochs = 100):

    # Input variables denoting the features and label data
    inputs_init = ct.input(shape=(num_channels, image_height, image_width))    
    inputs = ct.input(shape=(num_channels, image_height, image_width))
    labels = ct.input((num_classes))

    # apply model to input
    inputs_init_norm = (inputs_init - 127.5) / 127.5 # [-1, 1]    
    inputs_norm = (inputs - 127.5) / 127.5 # [-1, 1]

    z_init = m.build_model(inputs_init_norm, model, loss, first_run=True)
    z = m.build_model(inputs_norm, model, loss)

    # loss and metric
    ce = l.loss_function(inputs_norm, z, loss)
    pe = ct.relu(1.0) # dummy value to make reporting progress happy.

    # training config
    epoch_size              = 50000
    init_minibatch_size     = 100
    minibatch_size          = 12 if (model == 'pixelcnnpp') else 64
    sampling_minibatch_size = 16

    # Set learning parameters
    lr = 0.00001
    lr_decay = 0.6 #0.999995

    # Print progress
    progress_writers = [ct.logging.ProgressPrinter(tag='Training', freq=100, num_epochs=max_epochs)] # freq=100

    # trainer object
    learner = ct.learners.adam(z.parameters,
                               lr=ct.learning_rate_schedule(lr, unit=ct.UnitType.sample), 
                               momentum=ct.momentum_schedule(0.95), # Beta 1
                               unit_gain=False,
                               # use_mean_gradient=True,
                               variance_momentum=ct.momentum_schedule(0.9995), # Beta 2
                               # l1_regularization_weight = 0.001
                               l2_regularization_weight = 0.001
                               # gradient_clipping_threshold_per_sample=1
                               )
    trainer = ct.Trainer(z, (ce, pe), [learner], progress_writers)

    # define mapping from reader streams to network inputs
    input_map = {
        inputs: reader_train.streams.features,
        labels: reader_train.streams.labels
    }

    ct.logging.log_number_of_parameters(z); print()

    # perform model training
    prev_loss_before_sample = sys.float_info.max
    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        training_loss = 0

        # first_run
        if epoch == -1:
            reader_state = reader_train.get_checkpoint_state()
            data = reader_train.next_minibatch(min(init_minibatch_size, epoch_size), input_map=input_map)
            z_init.eval({inputs_init:data[inputs].asarray()})
            reader_train.restore_from_checkpoint(reader_state)

        sample_index = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size-sample_count), input_map=input_map)            
            trainer.train_minibatch({inputs:data[inputs].asarray()})

            sample_count  += trainer.previous_minibatch_sample_count
            training_loss += trainer.previous_minibatch_loss_average * trainer.previous_minibatch_sample_count

            # sample from the model
            if ((prev_loss_before_sample - trainer.previous_minibatch_loss_average) / prev_loss_before_sample) > 0.25:
                # sample(sampling_minibatch_size, z, inputs, nr_logistic_mix, loss, epoch, sample_index)
                prev_loss_before_sample = trainer.previous_minibatch_loss_average

            sample_index += 1

        sample(minibatch_size, z, inputs, nr_logistic_mix, loss, epoch, 0)

        trainer.summarize_training_progress()

        lr *= lr_decay
        learner.reset_learning_rate(ct.learning_rate_schedule(lr, unit=ct.UnitType.sample))

        # convert loss to bits/dim
        bits_per_dim = training_loss / (np.log(2.)*np.prod(image_shape)*sample_count)
        progress_writers[0].log("Bits per dimension: {}".format(bits_per_dim))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", 
                        "--model", 
                        type = str, 
                        help = "Specify which pixelcnn model to train: pixelcnn, pixelcnn2 or pixelcnnpp.", 
                        required = True)

    parser.add_argument("-l", 
                        "--loss", 
                        type = str,
                        help = "Specify which loss function to use: category or mixture", 
                        required = True)

    args = parser.parse_args()
    
    reader_train = create_reader(os.path.join(data_path, 'train_map.txt'), False)
    reader_test  = create_reader(os.path.join(data_path, 'test_map.txt'), False)

    train(reader_train, reader_test, args.model, args.loss)
