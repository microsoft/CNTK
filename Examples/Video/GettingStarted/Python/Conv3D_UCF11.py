﻿# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import sys
import os
import csv
import numpy as np
from random import randint

from PIL import Image
import imageio

import cntk as C
from cntk.logging import *
from cntk.debugging import set_computation_network_trace_level

# Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(abs_path, "..", "..", "DataSets", "UCF11")
model_path = os.path.join(abs_path, "Models")

# Define the reader for both training and evaluation action.
class VideoReader(object):
    '''
    A simple VideoReader: 
    It iterates through each video and select 16 frames as
    stacked numpy arrays.
    Similar to http://vlg.cs.dartmouth.edu/c3d/c3d_video.pdf
    '''
    def __init__(self, map_file, label_count, is_training, limit_epoch_size=sys.maxsize):
        '''
        Load video file paths and their corresponding labels.
        '''
        self.map_file        = map_file
        self.label_count     = label_count
        self.width           = 112
        self.height          = 112
        self.sequence_length = 16
        self.channel_count   = 3
        self.is_training     = is_training
        self.video_files     = []
        self.targets         = []
        self.batch_start     = 0

        map_file_dir = os.path.dirname(map_file)

        with open(map_file) as csv_file:
            data = csv.reader(csv_file)
            for row in data:
                self.video_files.append(os.path.join(map_file_dir, row[0]))
                target = [0.0] * self.label_count
                target[int(row[1])] = 1.0
                self.targets.append(target)

        self.indices = np.arange(len(self.video_files))
        if self.is_training:
            np.random.shuffle(self.indices)
        self.epoch_size = min(len(self.video_files), limit_epoch_size)

    def size(self):
        return self.epoch_size
            
    def has_more(self):
        if self.batch_start < self.size():
            return True
        return False

    def reset(self):
        if self.is_training:
            np.random.shuffle(self.indices)
        self.batch_start = 0

    def next_minibatch(self, batch_size):
        '''
        Return a mini batch of sequence frames and their corresponding ground truth.
        '''
        batch_end = min(self.batch_start + batch_size, self.size())
        current_batch_size = batch_end - self.batch_start
        if current_batch_size < 0:
            raise Exception('Reach the end of the training data.')

        inputs  = np.empty(shape=(current_batch_size, self.channel_count, self.sequence_length, self.height, self.width), dtype=np.float32)
        targets = np.empty(shape=(current_batch_size, self.label_count), dtype=np.float32)
        for idx in range(self.batch_start, batch_end):
            index = self.indices[idx]
            inputs[idx - self.batch_start, :, :, :, :] = self._select_features(self.video_files[index])
            targets[idx - self.batch_start, :]         = self.targets[index]

        self.batch_start += current_batch_size
        return inputs, targets, current_batch_size

    def _select_features(self, video_file):
        '''
        Select a sequence of frames from video_file and return them as
        a Tensor.
        '''
        video_reader = imageio.get_reader(video_file, 'ffmpeg')
        num_frames   = len(video_reader)
        if self.sequence_length > num_frames:
            raise ValueError('Sequence length {} is larger then the total number of frames {} in {}.'.format(self.sequence_length, num_frames, video_file))

        # select which sequence frames to use.
        step = 1
        expanded_sequence = self.sequence_length
        if num_frames > 2*self.sequence_length:
            step = 2
            expanded_sequence = 2*self.sequence_length

        seq_start = int(num_frames/2) - int(expanded_sequence/2)
        if self.is_training:
            seq_start = randint(0, num_frames - expanded_sequence)

        frame_range = [seq_start + step*i for i in range(self.sequence_length)]            
        video_frames = []
        for frame_index in frame_range:
            video_frames.append(self._read_frame(video_reader.get_data(frame_index)))
        
        return np.stack(video_frames, axis=1)

    def _read_frame(self, data):
        '''
        Based on http://vlg.cs.dartmouth.edu/c3d/c3d_video.pdf
        We resize the image to 128x171 first, then selecting a 112x112
        crop.
        '''
        if (self.width >= 171) or (self.height >= 128):
            raise ValueError("Target width need to be less than 171 and target height need to be less than 128.")
        
        image = Image.fromarray(data)
        image.thumbnail((171, 128), Image.ANTIALIAS)
        
        center_w = image.size[0] / 2
        center_h = image.size[1] / 2

        image = image.crop((center_w - self.width  / 2,
                            center_h - self.height / 2,
                            center_w + self.width  / 2,
                            center_h + self.height / 2))
        
        norm_image = np.array(image, dtype=np.float32)
        norm_image -= 127.5
        norm_image /= 127.5

        # (channel, height, width)
        return np.ascontiguousarray(np.transpose(norm_image, (2, 0, 1)))

# Creates and trains a feedforward classification model for UCF11 action videos
def conv3d_ucf11(train_reader, test_reader, max_epochs=30):
    # Replace 0 with 1 to get detailed log.
    set_computation_network_trace_level(0)

    # These values must match for both train and test reader.
    image_height       = train_reader.height
    image_width        = train_reader.width
    num_channels       = train_reader.channel_count
    sequence_length    = train_reader.sequence_length
    num_output_classes = train_reader.label_count

    # Input variables denoting the features and label data
    input_var = C.input_variable((num_channels, sequence_length, image_height, image_width), np.float32)
    label_var = C.input_variable(num_output_classes, np.float32)

    # Instantiate simple 3D Convolution network inspired by VGG network 
    # and http://vlg.cs.dartmouth.edu/c3d/c3d_video.pdf
    with C.default_options (activation=C.relu):
        z = C.layers.Sequential([
            C.layers.Convolution3D((3,3,3), 64, pad=True),
            C.layers.MaxPooling((1,2,2), (1,2,2)),
            C.layers.For(range(3), lambda i: [
                C.layers.Convolution3D((3,3,3), [96, 128, 128][i], pad=True),
                C.layers.Convolution3D((3,3,3), [96, 128, 128][i], pad=True),
                C.layers.MaxPooling((2,2,2), (2,2,2))
            ]),
            C.layers.For(range(2), lambda : [
                C.layers.Dense(1024), 
                C.layers.Dropout(0.5)
            ]),
            C.layers.Dense(num_output_classes, activation=None)
        ])(input_var)
    
    # loss and classification error.
    ce = C.cross_entropy_with_softmax(z, label_var)
    pe = C.classification_error(z, label_var)

    # training config
    train_epoch_size     = train_reader.size()
    train_minibatch_size = 2

    # Set learning parameters
    lr_per_sample          = [0.01]*10+[0.001]*10+[0.0001]
    lr_schedule            = C.learning_parameter_schedule_per_sample(lr_per_sample, epoch_size=train_epoch_size)
    momentum_per_sample = 0.9997558891748972
    mm_schedule            = C.momentum_schedule_per_sample([momentum_per_sample])

    # Instantiate the trainer object to drive the model training
    learner = C.momentum_sgd(z.parameters, lr_schedule, mm_schedule, True)
    progress_printer = ProgressPrinter(tag='Training', num_epochs=max_epochs)
    trainer = C.Trainer(z, (ce, pe), learner, progress_printer)

    log_number_of_parameters(z) ; print()

    # Get minibatches of images to train with and perform model training
    for epoch in range(max_epochs):       # loop over epochs
        train_reader.reset()

        while train_reader.has_more():
            videos, labels, current_minibatch = train_reader.next_minibatch(train_minibatch_size)
            trainer.train_minibatch({input_var : videos, label_var : labels})

        trainer.summarize_training_progress()

    # Test data for trained model
    epoch_size     = test_reader.size()
    test_minibatch_size = 2

    # process minibatches and evaluate the model
    metric_numer    = 0
    metric_denom    = 0
    minibatch_index = 0

    test_reader.reset()    
    while test_reader.has_more():
        videos, labels, current_minibatch = test_reader.next_minibatch(test_minibatch_size)
        # minibatch data to be trained with
        metric_numer += trainer.test_minibatch({input_var : videos, label_var : labels}) * current_minibatch
        metric_denom += current_minibatch
        # Keep track of the number of samples processed so far.
        minibatch_index += 1

    print("")
    print("Final Results: Minibatch[1-{}]: errs = {:0.2f}% * {}".format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom))
    print("")

    return metric_numer/metric_denom

if __name__=='__main__':
    num_output_classes = 11
    train_reader = VideoReader(os.path.join(data_path, 'train_map.csv'), num_output_classes, True)
    test_reader  = VideoReader(os.path.join(data_path, 'test_map.csv'), num_output_classes, False)
        
    conv3d_ucf11(train_reader, test_reader)
