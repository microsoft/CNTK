# Binary Convolution: Examples/Extensibility/BinaryConvolution

## Overview
This directory contains an implementation of training a binary convolution network, and a highly optimized native C++ implementation for fast evaluation of binary convolution models. The C++ binary convolution implementation utilizes the [Halide](http://halide-lang.org/) framework for making optimal use of multi-threading and vector instructions available on modern x86, x64 and ARM CPUs.

## Description
State-of-the-art convolutional neural networks typically require well over a billion floating point operations to classify a single image. Due to such high compute requirements, running convolution based classifiers on resource limited hardware such as Raspberry Pi or a smartphone, results in a very low evaluation framerate. There have been many efforts to speedup the inference of convolution networks and one recent line of inquiry offers a particularly impressive speedup; [BinaryConnect by Courbariaux et al.](https://arxiv.org/abs/1511.00363) The key idea here is to replace weights and activations of the convolution network with single bit values, without sacrificing the network's classification performance. By reducing to a single bit, floating point operations involved in the convolutions can be replaced with bitwise xnor operations. This allows up to 64 operations to be performed in a single clock cycle, by packing the bits of the weights and activations appropriately. While achieving a 64x speed up over hand optimized libraries like cblas is quite difficult, a solid 10x is easily achievable using a good optimization framework such as Halide, as illustrated here.

Single bit binarization essentially just takes the sign of each value, packs those inputs into chunks of 64, and then performs a series of xnor accumulates. While this sounds quite simple, it contains many operations that simply aren't available out-of-the-box in any mainstream deep learning framework. CNTK provides a powerful and performant extensibility facility, which allows to quickly implement custom tensor operators in Python or C++, that seamlessly operate in concert with built-in CNTK operators and training capabilities (including distributed training). This example contains a suite of network binarization implementations in the form of CNTK custom user-defined Functions. Here's a quick tour of the files and their contents:

| File | Description |
|:---------|:------------|
|[BinaryConvolveOp.h](../../../Source/Extensibility/BinaryConvolutionLib/BinaryConvolveOp.h)          |This file contains the fast C++ binary convolution implementation in form of a CNTK native user-defined Function. It calls into a Halide class (`HalideBinaryConvolve`) to perform the actual computations.
|[halide_binary_convolve.h](../../../Source/Extensibility/BinaryConvolutionLib/halide_binary_convolve.h) |The Halide definition of binarization and convolution kernels. Allows achieving good speedup with very little effort (as opposed to months of development efforts required for hand-optimized implementations); see http://halide-lang.org/
|[custom_convolution_ops.py](./custom_convolution_ops.py)                 |Python definitions of CNTK user-defined functions that emulate binarization. The purpose of these is not speedup but to allow for binary networks to be trained in a very simple way. They also serve as good examples of how to define CNTK custom user-defined functions purely in python. 
|[binary_convnet.py](./binary_convnet.py)                   |A driver script which defines a binary convolution network, trains it on the CIFAR10 dataset, and finally evaluates the model  using the optimized C++ binary convolution user-defined CNTK Function.

## Using this code

### Getting the data

CIFAR-10 dataset is not included in the CNTK distribution but can be easily downloaded and converted by following the instructions in [DataSets/CIFAR-10](../../Image/DataSets/CIFAR-10). We recommend you to keep the downloaded data in the respective folder while downloading, as the scripts in this folder assume that by default.

### Training and testing the binary convolution network

To run this code, invoke [binary_convnet.py](./binary_convnet.py), which creates a binary convolution network, and trains. Then, the code replaces the Python binary convolutions in the model with the native C++ binary convolution Functions, and evaluates the model on the CIFAR test-set.

## Editing the Halide Function
If you're interested in tweaking the binarization kernels defined in [halide_binary_convolve.h](../../../Source/Extensibility/BinaryConvolutionLib/halide_binary_convolve.h), you can simply change the code and build BinaryConvolution sub project to replace the libraries in your path.

## Defining your Own binary convolution model
Exploring other models with binarization is fairly easy using the functions provided. Simply define a model along the lines of `create_binary_convolution_model` in [binary_convnet.py](./binary_convnet.py)
 using the python user-defined Functions from custom_convolution_ops.py that fit your needs. The training and evaluation code in [binary_convnet.py](./binary_convnet.py)
 can then be used with the new model definition as shown.
