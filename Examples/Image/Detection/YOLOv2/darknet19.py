from __future__ import print_function
from cntk import *
from cntk.layers import *
from cntk.initializer import glorot_uniform
from cntk.io import MinibatchSource, ImageDeserializer, CTFDeserializer, StreamDefs, StreamDef
from cntk.io.transforms import scale
from cntk.layers import placeholder, constant
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_as_time_constant_schedule
from cntk.logging import log_number_of_parameters, ProgressPrinter
from cntk.logging.graph import find_by_name, plot
from PARAMETERS import *
from Cifar10Utils import *
import numpy as np
import os, sys
import cntk.io.transforms as xforms


def createFeatureExtractor(filter_multiplier=FIRSTLAYER_FILTERS):
    nfilters = filter_multiplier
    net = Sequential([
        BatchNormalization(name="feature_layer"),
        Convolution2D(filter_shape=(3,3),num_filters=nfilters,pad=True,activation=leaky_relu),
        BatchNormalization(name="stage_1"),
        MaxPooling(filter_shape=(2,2), strides=(2,2)),
        #Output: in_x/2 x in_y/2 x nfilters


        Convolution2D(filter_shape=(3, 3), num_filters=(nfilters * 2^1), pad=True, activation=leaky_relu),
        BatchNormalization(name="stage_2"),
        MaxPooling(filter_shape=(2, 2), strides=(2, 2)),
        #Ouptut: in_x/4 x in_y/4 x 2*nfilters


        Convolution2D(filter_shape=(3, 3), num_filters=(nfilters * 2^2), pad=True, activation=leaky_relu),
        BatchNormalization(),
        Convolution2D(filter_shape=(1, 1), num_filters=(nfilters * 2^1), pad=True, activation=leaky_relu), #Compression --> reduce number of filters by half
        BatchNormalization(),
        Convolution2D(filter_shape=(3, 3), num_filters=(nfilters * 2^2), pad=True, activation=leaky_relu),
        BatchNormalization(name="stage_3"),
        MaxPooling(filter_shape=(2, 2), strides=(2, 2)),
        #Output in_x/8 x in_y/8 x 4*nfilters


        Convolution2D(filter_shape=(3, 3), num_filters=(nfilters * 2 ^ 3), pad=True, activation=leaky_relu),
        BatchNormalization(),
        Convolution2D(filter_shape=(1, 1), num_filters=(nfilters * 2 ^ 2), pad=True, activation=leaky_relu), # Compression --> reduce number of filters by half
        BatchNormalization(),
        Convolution2D(filter_shape=(3, 3), num_filters=(nfilters * 2 ^ 3), pad=True, activation=leaky_relu),
        BatchNormalization(name="stage_4"),
        MaxPooling(filter_shape=(2, 2), strides=(2, 2)),
        # Output in_x/16 x in_y/16 x 8*nfilters


        Convolution2D(filter_shape=(3, 3), num_filters=(nfilters * 2 ^ 4), pad=True, activation=leaky_relu),
        BatchNormalization(),
        Convolution2D(filter_shape=(1, 1), num_filters=(nfilters * 2 ^ 3), pad=True, activation=leaky_relu), # Compression --> reduce number of filters by half
        BatchNormalization(),
        Convolution2D(filter_shape=(3, 3), num_filters=(nfilters * 2 ^ 4), pad=True, activation=leaky_relu),
        BatchNormalization(),
        Convolution2D(filter_shape=(1, 1), num_filters=(nfilters * 2 ^ 3), pad=True, activation=leaky_relu), # Compression --> reduce number of filters by half
        BatchNormalization(),
        Convolution2D(filter_shape=(3, 3), num_filters=(nfilters * 2 ^ 4), pad=True, activation=leaky_relu),
        BatchNormalization(name="stage_5"),
        MaxPooling(filter_shape=(2, 2), strides=(2, 2)),
        # Output in_x/32 x in_y/32 x 16*nfilters


        Convolution2D(filter_shape=(3, 3), num_filters=(nfilters * 2 ^ 5), pad=True, activation=leaky_relu),
        BatchNormalization(),
        Convolution2D(filter_shape=(1, 1), num_filters=(nfilters * 2 ^ 4), pad=True, activation=leaky_relu),# Compression --> reduce number of filters by half
        BatchNormalization(),
        Convolution2D(filter_shape=(3, 3), num_filters=(nfilters * 2 ^ 5), pad=True, activation=leaky_relu),
        BatchNormalization(),
        Convolution2D(filter_shape=(1, 1), num_filters=(nfilters * 2 ^ 4), pad=True, activation=leaky_relu),# Compression --> reduce number of filters by half
        BatchNormalization(),
        Convolution2D(filter_shape=(3, 3), num_filters=(nfilters * 2 ^ 5), pad=True, activation=leaky_relu, name="featurizer_output"),
        BatchNormalization()
        #Output in_x/32 x in_y/32 x 32*nfilters
    ],'featureExtractor_darknet19')

    return net

"""@Depricated; doesn't work!"""
def create_classification_head(nrOfClasses=OUTPUT_SIZE):

    net = Sequential([
        Convolution2D(filter_shape=(1,1), num_filters=nrOfClasses, pad=True, activation=identity,name="classifier_input"),
        GlobalAveragePooling(),
        reshape(x=3, shape=(10)),
        Activation(activation=softmax, name="classifier_output")

    ],"classifier_darknet19")

    return net

def put_classifier_on_featurizer(featurizer,nrOfClasses):
    return Sequential([
        reshape(x=Sequential([
            cntk.ops.minus(114),
            featurizer,
            Convolution2D(filter_shape=(1, 1), num_filters=nrOfClasses, pad=True, activation=identity,
                          name="classifier_input"),
            GlobalAveragePooling()
        ]), shape=(10)),
        Activation(activation=softmax, name="classifier_output")
    ],name="darknet19-classifier")

def create_classification_model(nrOfClasses, filter_mult=32):
    featurizer = createFeatureExtractor(filter_mult)
    #classifier = create_classification_head(nrOfClasses)

    return put_classifier_on_featurizer(featurizer, nrOfClasses)
    #return Sequential([featurizer, classifier], "darknet19")


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
    data_path = """D:\local\CNTK-2-0-rc1\cntk\Examples\Image\DataSets\CIFAR-10"""


    model = create_classification_model(num_classes)
    print("Created Model!")

    # train
    reader = create_reader(os.path.join(data_path, 'train_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), True)
    reader_test = create_reader(os.path.join(data_path, 'test_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'),
                                False)
    print("Created Readers!")

    train_model(reader, reader_test, model, max_epochs=80)

    # save and load (as an illustration)
   # model.save(path)
    save_model(model)

    # test
   # model = Function.load(path)
    reader = create_reader(os.path.join(data_path, 'test_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), False)
    evaluate(reader, model)

    save_model(model)