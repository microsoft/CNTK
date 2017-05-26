# Binary Convolution: Examples/Extensibility/BinaryConvolution

## Overview
This directory contains an implementation of training a binary convolution network, and a highly optimized native C++ implementation for fast evaluation of binary convolution models. The C++ binary convolution implementation utilizes the Halide (http://halide-lang.org/) framework for making optimal use of multi-threading and vector instructions available on modern x86, x64 and ARM CPUs.

## Description
State-of-the-art convolutional neural networks typically require well over a billion floating point operations to classify a single image. Due to such high compute requirements, running convolution based classifiers on resource limited hardware such as Raspberry-Pi or a smartphone, results in a very low evaluation framerate. There have been many efforts to speedup the inference of convolution networks and one recent line of inquiry offers a particularly impressive speedup; BinaryConnect by Courbariaus et Al (https://arxiv.org/abs/1511.00363). The key idea here is to replace weights and activations of the convolution network with single bit values, without sacrificing the network's classification performance. By reducing to a single bit, floating point operations involved in the convolutions can be replaced with bitwise xnor operations. This allows up to 64 operations to be performed in a single clock cycle, by packing the bits of the weights and activations appropriately. While achieving a 64x speed up over hand optimized libraries like cblas is quite difficult, a solid 10x is easily achievable using a good optimization framework such as Halide, as illustrated here.

Single bit binarization essentially just takes the sign of each value, packs those inputs into chunks of 64, and then performs a series of xnor accumulates. While this sounds quite simple, it contains many operations that simply aren't available out-of-the-box in any mainstream deep learning framework. CNTK provides a powerful and performance extensibility facility, which allows quickly implemeting custom tensor operators in python or C++, that seamlessly operate in concert with built-in CNTK operators and training capabilities (including distributed training). This example contains a suite of network binarization implementations in the form of CNTK custom user-defined Functions. Here's a quick tour of the files and their contents:

|[BinaryConvolveOp.h](./BinaryConvolutionLib/BinaryConvolveOp.h)          |This file contains the fast C++ binary convolution implementation in form of a CNTK native user-defined Function. It calls into a halide function (halide_convolve) to perform the actual computations.
|[halide_convolve.cpp](./BinaryConvolutionLib/halide/halide_convolve.cpp) |The Halide definition of binarization and convolution kernels. Allows achieving good speedup with very little effort (as opposed to months of development efforts required for hand-optimized implementations); see http://halide-lang.org/
|[halide_convolve.lib, halide_convolve.a, halide_convolve_nofeatures.a]   |The pre-built halide code that is used in the C++ binary convolution user-deined CNTK Function; there are 2 variants available for linux viz. halide_convolve_nofeatures.a which does not use SSE instructions and can be used on any x64 CPU and halide_convolve.a that uses SSE4 and AVX instructions and runs much faster, but needs a compatible modern CPU.
|[custom_convolution_ops.py](./custom_convolution_ops.py)                 |Python definitions of CNTK user-defined functions that emulate binarization. The purpose of these is not speedup but to allow for binary networks to be trained in a very simple way. They also serve as good examples of how to define CNTK custom user-defined functions purely in python. 
|[binary_convnet.py](./binary_convnet.py)                   |A driver script which defines a binary convolution network, trains it on the CIFAR10 dataset, and finally evaluates the model  using the optimized C++ binary convolution user-defined CNTK Function.

## Using this code

### Getting the data

CIFAR-10 dataset is not included in the CNTK distribution but can be easily downloaded and converted by following the instructions in [DataSets/CIFAR-10](../../Image/DataSets/CIFAR-10). We recommend you to keep the downloaded data in the respective folder while downloading, as the scripts in this folder assume that by default.

### Training and testing the binary convolution network

To run this code, invoke [binary_convnet.py](./binary_convnet.py), which creates a binary convolution network, and trains. Then, the code replaces the python binary convolutions in the model with the native C++ binary convolution Functions, and evaluates the model on the CIFAR test-set.

## Editing the Halide Function
If you're interested in tweaking the binarization kernels defined in halide_convolve.cpp, setup Halide by following the instructions at(https://github.com/halide/Halide/) and then build a new library with your changes, by simply running:
>> g++ -std=c++11 -I <Halide_Dir>/include/halide_convolve.cpp <Halide_Dir>/lib/libHalide.a -o halide_convolve -ldl -lpthread -ltinfo -lz
>> ./halide_convolve

Note that halide_convolve is currently set up to target the platform it's built on, but you can change it to target other things, even small ARM devices like the raspberry pi!

## Defining your Own binary convolution model
Exploring other models with binarization is fairly easy using the functions provided. Simply define a model along the lines of 'create_binary_convolution_model' in binary_convnet.py using the python user-defined Functions from custom_convolution_ops.py that fit your needs. The training and evaluation code in binary_convnet.py can then be used with the new model definition as shown.