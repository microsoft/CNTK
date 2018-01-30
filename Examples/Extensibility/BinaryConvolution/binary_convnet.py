# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import sys
import cntk as C

from custom_convolution_ops import *

# Instantiates a binary convolution layer
def BinaryConvolution(operand,
                      filter_shape,
                      num_filters=1,
                      channels = 1,
                      init=C.glorot_uniform(),
                      pad=False,
                      strides=1,
                      bias=True,
                      init_bias=0,
                      op_name='BinaryConvolution', name=''):
    """ arguments:
            operand: tensor to convolve
            filter_shape: tuple indicating filter size
            num_filters: number of filters to use 
            channels: number of incoming channels
            init: type of initialization to use for weights
    """
    kernel_shape = (num_filters, channels) + filter_shape
    W = C.parameter(shape=kernel_shape, init=init, name="filter")

    binary_convolve_operand_p = C.placeholder(operand.shape, operand.dynamic_axes, name="operand")
    binary_convolve = C.convolution(CustomMultibit(W, 1), CustomMultibit(binary_convolve_operand_p, 1), auto_padding=[False, pad, pad], strides=[strides])
    r = C.as_block(binary_convolve, [(binary_convolve_operand_p, operand)], 'binary_convolve')

    bias_shape = (num_filters, 1, 1)
    b = C.parameter(shape=bias_shape, init=init_bias, name="bias")
    r = r + b

    # apply learnable param relu
    P = C.parameter(shape=r.shape, init=init, name="prelu")
    r = C.param_relu(P, r)
    return r

# Create the binary convolution network for training.
def create_binary_convolution_model():

    # Input variables denoting the features and label data
    feature_var = C.input((num_channels, image_height, image_width))
    label_var = C.input((num_classes))

    # apply model to input
    scaled_input = C.element_times(C.constant(0.00390625), feature_var)

    # first layer is ok to be full precision
    z = C.layers.Convolution((3, 3), 64, pad=True, activation=C.relu)(scaled_input)
    z = C.layers.MaxPooling((3,3), strides=(2,2))(z)

    z = C.layers.BatchNormalization(map_rank=1)(z)
    z = BinaryConvolution(z, (3,3), 128, channels=64, pad=True)
    z = C.layers.MaxPooling((3,3), strides=(2,2))(z)

    z = C.layers.BatchNormalization(map_rank=1)(z)
    z = BinaryConvolution(z, (3,3), 128, channels=128, pad=True)
    z = C.layers.MaxPooling((3,3), strides=(2,2))(z)

    z = C.layers.BatchNormalization(map_rank=1)(z)
    z = BinaryConvolution(z, (1,1), num_classes, channels=128, pad=True)
    z = C.layers.AveragePooling((z.shape[1], z.shape[2]))(z)
    z = C.reshape(z, (num_classes,))

    # Add binary regularization (ala Gang Hua)
    weight_sum = C.constant(0)
    for p in z.parameters:
        if (p.name == "filter"):
            weight_sum = C.plus(weight_sum, C.reduce_sum(C.minus(1, C.square(p))))
    bin_reg = C.element_times(.000005, weight_sum)

    # After the last layer, we need to apply a learnable scale
    SP = C.parameter(shape=z.shape, init=0.001)
    z = C.element_times(z, SP)

    # loss and metric
    ce = C.cross_entropy_with_softmax(z, label_var)
    ce = C.plus(ce, bin_reg)
    pe = C.classification_error(z, label_var)

    return C.combine([z, ce, pe])

# Clones a binary convolution network, sharing the original parameters  but substitutes the
# python 'binary_convolve' Function instances used during training, faster C++ NativeBinaryConvolveFunction
# instances that uses optimized binary convolution implementations generated using the Halide framework
def clone_with_native_binary_convolutions(model):
    # using a different name to avoid conflict with netopt package. 
    # netopt uses NativeBinaryConvolveFunction as the name.
    ops.register_native_user_function('BinaryConvolutionFunction', 'Cntk.BinaryConvolution-' + C.__version__.rstrip('+'), 'CreateBinaryConvolveFunction')
    filter = lambda x : type(x) == C.Function and x.root_function.op_name == 'binary_convolve'

    def converter(x):
        # TODO: The attributes should be read from x instead of hardcoded values
        attributes = {'stride' : 1, 'padding' : True, 'size' : x.inputs[0].shape[-1], 'w' : x.inputs[1].shape[-2], 'h'
                : x.inputs[1].shape[-1], 'channels' : x.inputs[1].shape[0], 'filters' : x.inputs[0].shape[0]}
        return ops.native_user_function('BinaryConvolutionFunction', list(x.inputs), attributes, 'native_binary_convolve')

    return C.misc.convert(model, filter, converter)

def get_z_and_criterion(combined_model):
    return (C.combine([combined_model.outputs[0].owner]), C.combine([combined_model.outputs[1].owner, combined_model.outputs[2].owner]))

# Import training and evaluation routines from ConvNet_CIFAR10_DataAug
abs_path = os.path.dirname(os.path.abspath(__file__))
custom_convolution_ops_dir = os.path.join(abs_path, "..", "..", "Image", "Classification", "ConvNet", "Python")
sys.path.append(custom_convolution_ops_dir)

from ConvNet_CIFAR10_DataAug import *

############################# 
# main function boilerplate #
#############################

if __name__=='__main__':
    model = create_binary_convolution_model()
    z, criterion = get_z_and_criterion(model)
    reader_train = create_reader(os.path.join(data_path, 'train_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), True)
    train_model(reader_train, z, criterion, max_epochs=80)

    # save and load (as an illustration)
    model_path = data_path + "/model.cmf"
    model.save(model_path)

    # We use the NativeBinaryConvolveFunction for testing the model, which currently only runs on the CPU
    eval_device = C.cpu()
    model = Function.load(model_path, device=eval_device)

    # For testing, replace all python binary convolution user-functions with the fast Halide generated
    # NativeBinaryConvolveFunction. Note, the NativeBinaryConvolveFunction currently only supports eval,
    # and is thus not used for training.
    model_with_native_binary_convolutions = clone_with_native_binary_convolutions(model)
    _, criterion = get_z_and_criterion(model_with_native_binary_convolutions)

    reader_test = create_reader(os.path.join(data_path, 'test_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), False)

    # TODO: The NativeBinaryConvolveFunction can currently only process one image at a time
    evaluate(reader_test, criterion, device=eval_device, minibatch_size=1, max_samples=1000)
