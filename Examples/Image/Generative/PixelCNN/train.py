from __future__ import print_function
import os
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
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')
    ]
    # deserializer
    return ct.io.MinibatchSource(ct.io.ImageDeserializer(map_file, ct.io.StreamDefs(
        features = ct.io.StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
        labels   = ct.io.StreamDef(field='label', shape=num_classes))),   # and second as 'label'
        randomize=is_training)

def train(reader_train, reader_test, model, loss, epoch_size = 50000, max_epochs = 100):

    # Input variables denoting the features and label data
    input_var  = ct.input_variable((num_channels, image_height, image_width))
    target_var = ct.input_variable(shape=(256, num_channels*image_height*image_width)) if (loss == 'category') else ct.input_variable(shape=(num_channels, image_height, image_width))
    label_var  = ct.input_variable((num_classes))

    # apply model to input
    input_norm = (input_var - 127.5) / 127.5 # [-1, 1]
    z = m.build_model(input_norm, model, loss)

    # loss and metric
    ce = l.loss_function(input_norm, target_var, z, loss)
    pe = ct.relu(1.0) # dummy value to make reporting progress happy.

    # training config
    epoch_size     = 50000    
    minibatch_size = 12 if (model == 'pixelcnnpp') else 64


    # Set learning parameters
    lr_init          = 0.00001
    lr_decay         = 0.999995
    lr_per_sample    = lr_init
    lr_schedule      = ct.learning_rate_schedule(lr_per_sample, unit=ct.UnitType.sample)
    mm_time_constant = 4096
    mm_schedule      = ct.momentum_as_time_constant_schedule(mm_time_constant)

    # Print progress
    progress_writers = [ct.logging.ProgressPrinter(tag='Training', freq=100, num_epochs=max_epochs)] # freq=100

    # trainer object
    learner = ct.learners.adam(z.parameters, 
                               lr=lr_schedule, 
                               momentum=mm_schedule,
                               unit_gain=False,
                               # l1_regularization_weight = 0.001
                               # l2_regularization_weight = 0.001,
                               gradient_clipping_threshold_per_sample=100
                               )
    trainer = ct.Trainer(z, (ce, pe), [learner], progress_writers)

    # define mapping from reader streams to network inputs
    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    ct.logging.log_number_of_parameters(z); print()

    # perform model training
    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        training_loss = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            t0 = time.perf_counter()
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size-sample_count), input_map=input_map) # fetch minibatch.
            t1 = time.perf_counter()

            if loss == 'category':
                # One hot: 256, 3*32*32
                image  = np.asarray(data[input_var].value, dtype=int).flatten()
                target = np.zeros((256,) + image.shape)
                target[image, np.arange(image.size)] = 1
                target = np.ascontiguousarray(np.reshape(target, (-1, 1, 256, num_channels*image_height*image_width)))
                trainer.train_minibatch({input_var:data[input_var].value, target_var:target})
            else:
                trainer.train_minibatch({input_var:data[input_var].value})
                # lr_per_sample *= lr_decay
                # learner.reset_learning_rate(ct.learning_rate_schedule(lr_per_sample, unit=ct.UnitType.sample))

            t2 = time.perf_counter()

            sample_count  += trainer.previous_minibatch_sample_count
            training_loss += trainer.previous_minibatch_loss_average * trainer.previous_minibatch_sample_count

        # sample from the model
        t3 = time.perf_counter()
        if loss == 'mixture':
            x_gen = np.zeros(image_shape, dtype=np.float32)
            for y in range(image_height):
                for x in range(image_width):
                    new_x_gen    = z.eval({input_var:[x_gen]})
                    new_x_gen_np = np.asarray(sp.np_sample_from_discretized_mix_logistic(new_x_gen[0], nr_logistic_mix))
                    x_gen[0,y,x] = new_x_gen_np[0,y,x]
                    x_gen[1,y,x] = new_x_gen_np[1,y,x]
                    x_gen[2,y,x] = new_x_gen_np[2,y,x]

            norm_image  = sp.scale_to_unit_interval(np.ascontiguousarray(np.transpose(x_gen, (1, 2, 0)), dtype=np.float32)) # [0,1]
            norm_image *= 255.0
            image = Image.fromarray(np.asarray(norm_image, dtype=np.uint8))
            image.save("image_{}.png".format(epoch))
        t4 = time.perf_counter()

        trainer.summarize_training_progress()

        # convert loss to bits/dim
        bits_per_dim = training_loss/(np.log(2.)*np.prod((image_height,image_width,num_channels))*sample_count)
        print("Bits per dimension: {}".format(bits_per_dim))

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
