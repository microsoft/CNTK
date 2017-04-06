# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
from cntk import *
from cntk.layers import *
from Cifar10Utils import *
import numpy as np
import os, sys
import cntk.io.transforms as xforms


# Creates the feature extractor shared by the classifier (Darknet19) and the Detector (YOLOv2)
def createFeatureExtractor(filter_multiplier=32):
    nfilters = filter_multiplier
    net = Sequential([
        BatchNormalization(name="feature_layer"),
        Convolution2D(filter_shape=(3,3),num_filters=nfilters,pad=True,activation=leaky_relu),
        BatchNormalization(),
        MaxPooling(filter_shape=(2,2), strides=(2,2)),
        #Output: in_x/2 x in_y/2 x nfilters


        Convolution2D(filter_shape=(3, 3), num_filters=(nfilters * 2**1), pad=True, activation=leaky_relu, name="stage_1"),
        BatchNormalization(),
        MaxPooling(filter_shape=(2, 2), strides=(2, 2)),
        #Ouptut: in_x/4 x in_y/4 x 2*nfilters


        Convolution2D(filter_shape=(3, 3), num_filters=(nfilters * 2**2), pad=True, activation=leaky_relu, name="stage_2"),
        BatchNormalization(),
        Convolution2D(filter_shape=(1, 1), num_filters=(nfilters * 2**1), pad=True, activation=leaky_relu),
        BatchNormalization(),
        Convolution2D(filter_shape=(3, 3), num_filters=(nfilters * 2**2), pad=True, activation=leaky_relu),
        BatchNormalization(),
        MaxPooling(filter_shape=(2, 2), strides=(2, 2)),
        #Output in_x/8 x in_y/8 x 4*nfilters


        Convolution2D(filter_shape=(3, 3), num_filters=(nfilters * 2**3), pad=True, activation=leaky_relu, name="stage_3"),
        BatchNormalization(),
        Convolution2D(filter_shape=(1, 1), num_filters=(nfilters * 2**2), pad=True, activation=leaky_relu),
        BatchNormalization(),
        Convolution2D(filter_shape=(3, 3), num_filters=(nfilters * 2**3), pad=True, activation=leaky_relu),
        BatchNormalization(),
        MaxPooling(filter_shape=(2, 2), strides=(2, 2)),
        # Output in_x/16 x in_y/16 x 8*nfilters


        Convolution2D(filter_shape=(3, 3), num_filters=(nfilters * 2**4), pad=True, activation=leaky_relu, name="stage_4"),
        BatchNormalization(),
        Convolution2D(filter_shape=(1, 1), num_filters=(nfilters * 2**3), pad=True, activation=leaky_relu),
        BatchNormalization(),
        Convolution2D(filter_shape=(3, 3), num_filters=(nfilters * 2**4), pad=True, activation=leaky_relu),
        BatchNormalization(),
        Convolution2D(filter_shape=(1, 1), num_filters=(nfilters * 2**3), pad=True, activation=leaky_relu),
        BatchNormalization(),
        Convolution2D(filter_shape=(3, 3), num_filters=(nfilters * 2**4), pad=True, activation=leaky_relu),
        BatchNormalization(),
        MaxPooling(filter_shape=(2, 2), strides=(2, 2)),
        # Output in_x/32 x in_y/32 x 16*nfilters


        Convolution2D(filter_shape=(3, 3), num_filters=(nfilters * 2**5), pad=True, activation=leaky_relu, name="stage_5"),
        BatchNormalization(),
        Convolution2D(filter_shape=(1, 1), num_filters=(nfilters * 2**4), pad=True, activation=leaky_relu),
        BatchNormalization(),
        Convolution2D(filter_shape=(3, 3), num_filters=(nfilters * 2**5), pad=True, activation=leaky_relu),
        BatchNormalization(),
        Convolution2D(filter_shape=(1, 1), num_filters=(nfilters * 2**4), pad=True, activation=leaky_relu),
        BatchNormalization(),
        Convolution2D(filter_shape=(3, 3), num_filters=(nfilters * 2**5), pad=True, activation=leaky_relu),
        BatchNormalization(name="featureExtractor_output")
        #Output in_x/32 x in_y/32 x 32*nfilters
    ],'featureExtractor_darknet19')

    return net

	
# Puts a classifier end to any feature extractor
def put_classifier_on_feature_extractor(featureExtractor,nrOfClasses):
    return Sequential([
        reshape(x=Sequential([
            [lambda x: x - 114],
            featureExtractor,
            Convolution2D(filter_shape=(1, 1), num_filters=nrOfClasses, pad=True, activation=identity,
                          name="classifier_input"),
            GlobalAveragePooling()
        ]), shape=(10)),
        Activation(activation=softmax, name="classifier_output")
    ],name="darknet19-classifier")


# Creates a Darknet19 classifier
def create_classification_model(nrOfClasses, filter_mult=32):
    featureExtractor = createFeatureExtractor(filter_mult)
    return put_classifier_on_feature_extractor(featureExtractor, nrOfClasses)


# Saves a model to the Output folder. If the models are already existing a ascending number is assigned to the model.
def save_model(model, name="darknet19"):
    abs_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(abs_path, "Output", name, ".model")
    if os.path.exists(model_path):
        i = 2
        while(True):
            model_path = os.path.join(abs_path, "Output", name,"(",i,")", ".model")
            if os.path.exists(model_path):
                i += 1
            else:
                break
    model.save(model_path)
    print("Stored model " + name + " to " + model_path)


########################################################################################################################
#   Main                                                                                                               #
########################################################################################################################

if __name__=='__main__':
    data_path =os.path.join("..","..","DataSets","CIFAR-10")

    # create
    model = create_classification_model(num_classes)
    print("Created Model!")

    # train
    reader = create_reader(os.path.join(data_path, 'train_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), True)
    reader_test = create_reader(os.path.join(data_path, 'test_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'),
                                False)
    print("Created Readers!")

    train_model(reader, reader_test, model, max_epochs=80)

    # save
    save_model(model)

    # test
    reader = create_reader(os.path.join(data_path, 'test_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), False)
    evaluate(reader, model)

    #save_model(model)
    print("Done!")
