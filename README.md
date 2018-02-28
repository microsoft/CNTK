[![Join the chat at https://gitter.im/Microsoft/CNTK](https://badges.gitter.im/Microsoft/CNTK.svg)](https://gitter.im/Microsoft/CNTK?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

| **Windows** | **Linux** |
|-------------|---------------|
| [![Build Status](https://aiinfra.visualstudio.com/_apis/public/build/definitions/a95b3960-90bb-440b-bd18-d3ec5d1cf8c3/126/badge)](https://cntk.ai/nightly-windows.html) | [![Build Status](https://aiinfra.visualstudio.com/_apis/public/build/definitions/a95b3960-90bb-440b-bd18-d3ec5d1cf8c3/127/badge)](https://cntk.ai/nightly-linux.html) |

## Latest news

***2018-02-28.*** CNTK supports nightly build

If you prefer to use latest CNTK bits from master, use one of the CNTK nightly package.
* [Nightly packages for Windows](https://cntk.ai/nightly-windows.html)
* [Nightly packages for Linux](https://cntk.ai/nightly-linux.html)

Alternatively, you can also click corresponding build badge to land to nightly build page.

***2018-01-31.* CNTK 2.4**

Highlights:
* Moved to CUDA9, cuDNN 7 and Visual Studio 2017.
* Removed Python 3.4 support.
* Added Volta GPU and FP16 support.
* Better ONNX support.
* CPU perf improvement.
* More OPs.

OPs
* ``top_k`` operation: in the forward pass it computes the top (largest) k values and corresponding indices along the specified axis. In the backward pass the gradient is scattered to the top k elements (an element not in the top k gets a zero gradient).
* ``gather`` operation now supports an axis argument
* ``squeeze`` and ``expand_dims`` operations for easily removing and adding singleton axes
* ``zeros_like`` and ``ones_like`` operations. In many situations you can just rely on CNTK correctly broadcasting a simple 0 or 1 but sometimes you need the actual tensor.
* ``depth_to_space``: Rearranges elements in the input tensor from the depth dimension into spatial blocks. Typical use of this operation is for implementing sub-pixel convolution for some image super-resolution models.
* ``space_to_depth``: Rearranges elements in the input tensor from the spatial dimensions to the depth dimension. It is largely the inverse of DepthToSpace.
* ``sum`` operation: Create a new Function instance that computes element-wise sum of input tensors.
* ``softsign`` operation: Create a new Function instance that computes the element-wise softsign of a input tensor.
* ``asinh`` operation: Create a new Function instance that computes the element-wise asinh of a input tensor.
* ``log_softmax`` operation: Create a new Function instance that computes the logsoftmax normalized values of a input tensor.
* ``hard_sigmoid`` operation: Create a new Function instance that computes the hard_sigmoid normalized values of a input tensor.
* ``element_and``, ``element_not``, ``element_or``, ``element_xor`` element-wise logic operations
* ``reduce_l1`` operation: Computes the L1 norm of the input tensor's element along the provided axes.
* ``reduce_l2`` operation: Computes the L2 norm of the input tensor's element along the provided axes..
* ``reduce_sum_square`` operation: Computes the sum square of the input tensor's element along the provided axes.
* ``image_scaler`` operation: Alteration of image by scaling its individual values.

ONNX
* There have been several improvements to ONNX support in CNTK.
* Updates
  * Updated ONNX ``Reshape`` op to handle ``InferredDimension``.
  * Adding ``producer_name`` and ``producer_version`` fields to ONNX models.
  * Handling the case when neither ``auto_pad`` nor ``pads`` atrribute is specified in ONNX ``Conv`` op.
* Bug fixes
  * Fixed bug in ONNX ``Pooling`` op serialization
  * Bug fix to create ONNX ``InputVariable`` with only one batch axis.
  * Bug fixes and updates to implementation of ONNX ``Transpose`` op to match updated spec.
  * Bug fixes and updates to implementation of ONNX ``Conv``, ``ConvTranspose``, and ``Pooling`` ops to match updated spec.

Operators
* Group convolution
  * Fixed bug in group convolution. Output of CNTK ``Convolution`` op will change for groups > 1. More optimized implementation of group convolution is expected in the next release.
  * Better error reporting for group convolution in ``Convolution`` layer.

Halide Binary Convolution
- The CNTK build can now use optional [Halide](http://halide-lang.org/) libraries to build ``Cntk.BinaryConvolution.so/dll`` library that can be used with the ``netopt`` module. The library contains optimized binary convolution operators that perform better than the python based binarized convolution operators. To enable Halide in the build, please download [Halide release](https://github.com/halide/Halide/releases) and set ``HALIDE_PATH`` environment varibale before starting a build. In Linux, you can use ``./configure --with-halide[=directory]`` to enable it. For more information on how to use this feature, please refer to [How_to_use_network_optimization](https://github.com/Microsoft/CNTK/blob/master/Manual/Manual_How_to_use_network_optimizations.ipynb).

See more in the [Release Notes](https://docs.microsoft.com/en-us/cognitive-toolkit/ReleaseNotes/CNTK_2_4_Release_Notes).
Get the Release from the [CNTK Releases page](https://github.com/Microsoft/CNTK/releases).


***2018-01-22.*** CNTK support for CUDA 9

CNTK now supports CUDA 9/cuDNN 7. This requires an update to build environment to Ubuntu 16/GCC 5 for Linux, and Visual Studio 2017/VCTools 14.11 for Windows. With CUDA 9, CNTK also added a preview for 16-bit floating point (a.k.a FP16) computation.

Please check out the example of FP16 in ResNet50 [here](./Examples/Image/Classification/ResNet/Python/TrainResNet_ImageNet_Distributed.py)

Notes on FP16 preview:
* FP16 implementation on CPU is not optimized, and it's not supposed to be used in CPU inference directly. User needs to convert the model to 32-bit floating point before running on CPU.
* Loss/Criterion for FP16 training needs to be 32bit for accumulation without overflow, using cast function. Please check the example above.
* Readers do not have FP16 output unless using numpy to feed data, cast from FP32 to FP16 is needed. Please check the example above.
* FP16 gradient aggregation is currently only implemented on GPU using NCCL2. Distributed training with FP16 with MPI is not supported.
* FP16 math is a subset of current FP32 implementation. Some model may get Feature Not Implemented exception using FP16.
* FP16 is currently not supported in BrainScript. Please use Python for FP16.

To setup build and runtime environment on Windows:
* Install [Visual Studio 2017](https://www.visualstudio.com/downloads/) with following workloads and components. From command line (use Community version installer as example):
    vs_community.exe --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Workload.ManagedDesktop --add Microsoft.VisualStudio.Workload.Universal --add Microsoft.Component.PythonTools --add Microsoft.VisualStudio.Component.VC.Tools.14.11
* Install [NVidia CUDA 9](https://developer.nvidia.com/cuda-90-download-archive?target_os=Windows&target_arch=x86_64)
* From PowerShell, run:
    [DevInstall.ps1](./Tools/devInstall/Windows/DevInstall.ps1)
* Start VCTools 14.11 command line, run:
    cmd /k "%VS2017INSTALLDIR%\VC\Auxiliary\Build\vcvarsall.bat" x64 --vcvars_ver=14.11
* Open [CNTK.sln](./CNTK.sln) from the VCTools 14.11 command line. Note that starting CNTK.sln other than VCTools 14.11 command line, would causes CUDA 9 [build error](https://developercommunity.visualstudio.com/content/problem/163758/vs-2017-155-doesnt-support-cuda-9.html).

To setup build and runtime environment on Linux using docker, please build Unbuntu 16.04 docker image using Dockerfiles [here](./Tools/docker). For other Linux systems, please refer to the Dockerfiles to setup dependent libraries for CNTK.

***2017-12-05.* CNTK 2.3.1**
Release of Cognitive Toolkit v.2.3.1.

CNTK support for ONNX format is now out of preview mode.

If you want to try ONNX, you can build from master or `pip install` one of the below wheels that matches your Python environment.

For Windows CPU-Only:
* Python 2.7: https://cntk.ai/PythonWheel/CPU-Only/cntk-2.3.1-cp27-cp27m-win_amd64.whl
* Python 3.4: https://cntk.ai/PythonWheel/CPU-Only/cntk-2.3.1-cp34-cp34m-win_amd64.whl
* Python 3.5: https://cntk.ai/PythonWheel/CPU-Only/cntk-2.3.1-cp35-cp35m-win_amd64.whl
* Python 3.6: https://cntk.ai/PythonWheel/CPU-Only/cntk-2.3.1-cp36-cp36m-win_amd64.whl

For Windows GPU:
* Python 2.7: https://cntk.ai/PythonWheel/GPU/cntk-2.3.1-cp27-cp27m-win_amd64.whl
* Python 3.4: https://cntk.ai/PythonWheel/GPU/cntk-2.3.1-cp34-cp34m-win_amd64.whl
* Python 3.5: https://cntk.ai/PythonWheel/GPU/cntk-2.3.1-cp35-cp35m-win_amd64.whl
* Python 3.6: https://cntk.ai/PythonWheel/GPU/cntk-2.3.1-cp36-cp36m-win_amd64.whl

For Windows GPU-1bit-SGD:
* Python 2.7: https://cntk.ai/PythonWheel/GPU-1bit-SGD/cntk-2.3.1-cp27-cp27m-win_amd64.whl
* Python 3.4: https://cntk.ai/PythonWheel/GPU-1bit-SGD/cntk-2.3.1-cp34-cp34m-win_amd64.whl
* Python 3.5: https://cntk.ai/PythonWheel/GPU-1bit-SGD/cntk-2.3.1-cp35-cp35m-win_amd64.whl
* Python 3.6: https://cntk.ai/PythonWheel/GPU-1bit-SGD/cntk-2.3.1-cp36-cp36m-win_amd64.whl

Linux CPU-Only:
* Python 2.7: https://cntk.ai/PythonWheel/CPU-Only/cntk-2.3.1-cp27-cp27mu-linux_x86_64.whl
* Python 3.4: https://cntk.ai/PythonWheel/CPU-Only/cntk-2.3.1-cp34-cp34m-linux_x86_64.whl
* Python 3.5: https://cntk.ai/PythonWheel/CPU-Only/cntk-2.3.1-cp35-cp35m-linux_x86_64.whl
* Python 3.6: https://cntk.ai/PythonWheel/CPU-Only/cntk-2.3.1-cp36-cp36m-linux_x86_64.whl

Linux GPU:
* Python 2.7: https://cntk.ai/PythonWheel/GPU/cntk-2.3.1-cp27-cp27mu-linux_x86_64.whl
* Python 3.4: https://cntk.ai/PythonWheel/GPU/cntk-2.3.1-cp34-cp34m-linux_x86_64.whl
* Python 3.5: https://cntk.ai/PythonWheel/GPU/cntk-2.3.1-cp35-cp35m-linux_x86_64.whl
* Python 3.6: https://cntk.ai/PythonWheel/GPU/cntk-2.3.1-cp36-cp36m-linux_x86_64.whl

Linux GPU-1bit-SGD:
* Python 2.7: https://cntk.ai/PythonWheel/GPU-1bit-SGD/cntk-2.3.1-cp27-cp27mu-linux_x86_64.whl
* Python 3.4: https://cntk.ai/PythonWheel/GPU-1bit-SGD/cntk-2.3.1-cp34-cp34m-linux_x86_64.whl
* Python 3.5: https://cntk.ai/PythonWheel/GPU-1bit-SGD/cntk-2.3.1-cp35-cp35m-linux_x86_64.whl
* Python 3.6: https://cntk.ai/PythonWheel/GPU-1bit-SGD/cntk-2.3.1-cp36-cp36m-linux_x86_64.whl

You can also try one of the below NuGet package.
* [CNTK, CPU-Only Build](https://www.nuget.org/packages/CNTK.CPUOnly/2.3.1)
* [CNTK, GPU Build](https://www.nuget.org/packages/CNTK.GPU/2.3.1)
* [CNTK, UWP CPU-Only Build](http://www.nuget.org/packages/CNTK.UWP.CPUOnly/2.3.1)
* [CNTK CPU-only Model Evaluation Libraries (MKL based)](http://www.nuget.org/packages/Microsoft.Research.CNTK.CpuEval-mkl/2.3.1)


***2017-11-22.* CNTK 2.3**
Release of Cognitive Toolkit v.2.3.

Highlights:
* Better ONNX support.
* Switched to NCCL2 for better performance in distributed training.
* Improved C# API.
* OpenCV is not required to install CNTK, it is only required for Tensorboard Image feature and image reader.
* Various performance improvement.
* Added Network Optimization API.
* Faster Adadelta for sparse.

See more in the [Release Notes](https://docs.microsoft.com/en-us/cognitive-toolkit/ReleaseNotes/CNTK_2_3_Release_Notes).  
Get the Release from the [CNTK Releases page](https://github.com/Microsoft/CNTK/releases).

***2017-11-10.*** Switch from CNTKCustomMKL to Intel MKLML. MKLML is released with [Intel MKL-DNN](https://github.com/01org/mkl-dnn/releases) as a trimmed version of Intel MKL for MKL-DNN. To set it up:

On Linux:

    sudo mkdir /usr/local/mklml
    sudo wget https://github.com/01org/mkl-dnn/releases/download/v0.11/mklml_lnx_2018.0.1.20171007.tgz
    sudo tar -xzf mklml_lnx_2018.0.1.20171007.tgz -C /usr/local/mklml

On Windows:

    Create a directory on your machine to hold MKLML, e.g. mkdir c:\local\mklml
    Download the file [mklml_win_2018.0.1.20171007.zip](https://github.com/01org/mkl-dnn/releases/download/v0.11/mklml_win_2018.0.1.20171007.zip).
    Unzip it into your MKLML path, creating a versioned sub directory within.
    Set the environment variable `MKLML_PATH` to the versioned sub directory, e.g. setx MKLML_PATH c:\local\mklml\mklml_win_2018.0.1.20171007

***2017-10-10.*** Preview: CNTK ONNX Format Support
Update CNTK to support load and save ONNX format from https://github.com/onnx/onnx, please try it and provide feedback. We only support ONNX OPs. This is a preview, and we expect a breaking change in the future.

* Support loading a model saved in ONNX format.
* Support saving a model in ONNX format, not all CNTK models are currently supported. Only a subset of CNTK models are supported and no RNN. We will add more in the future.

To load an ONNX model, simply specify the format parameter for the load function.
```
import cntk as C

C.Function.load(<path of your ONNX model>, format=C.ModelFormat.ONNX)
```

To save a CNTK graph as ONNX model, simply specify the format in the save function.

```
import cntk as C

x = C.input_variable(<input shape>)
z = create_model(x)
z.save(<path of where to save your ONNX model>, format=C.ModelFormat.ONNX)
```

If you want to try ONNX, you can build from master or `pip install` one of the below wheel that matches you Python environment.

For Windows CPU-Only:
* Python 2.7: https://cntk.ai/PythonWheel/CPU-Only/cntk-2.3-Pre-cp27-cp27m-win_amd64.whl
* Python 3.4: https://cntk.ai/PythonWheel/CPU-Only/cntk-2.3-Pre-cp34-cp34m-win_amd64.whl
* Python 3.5: https://cntk.ai/PythonWheel/CPU-Only/cntk-2.3-Pre-cp35-cp35m-win_amd64.whl
* Python 3.6: https://cntk.ai/PythonWheel/CPU-Only/cntk-2.3-Pre-cp36-cp36m-win_amd64.whl

For Windows GPU:
* Python 2.7: https://cntk.ai/PythonWheel/GPU/cntk-2.3-Pre-cp27-cp27m-win_amd64.whl
* Python 3.4: https://cntk.ai/PythonWheel/GPU/cntk-2.3-Pre-cp34-cp34m-win_amd64.whl
* Python 3.5: https://cntk.ai/PythonWheel/GPU/cntk-2.3-Pre-cp35-cp35m-win_amd64.whl
* Python 3.6: https://cntk.ai/PythonWheel/GPU/cntk-2.3-Pre-cp36-cp36m-win_amd64.whl

Linux CPU-Only:
* Python 2.7: https://cntk.ai/PythonWheel/CPU-Only/cntk-2.3-Pre-cp27-cp27mu-linux_x86_64.whl
* Python 3.4: https://cntk.ai/PythonWheel/CPU-Only/cntk-2.3-Pre-cp34-cp34m-linux_x86_64.whl
* Python 3.5: https://cntk.ai/PythonWheel/CPU-Only/cntk-2.3-Pre-cp35-cp35m-linux_x86_64.whl
* Python 3.6: https://cntk.ai/PythonWheel/CPU-Only/cntk-2.3-Pre-cp36-cp36m-linux_x86_64.whl

Linux GPU:
* Python 2.7: https://cntk.ai/PythonWheel/GPU/cntk-2.3-Pre-cp27-cp27mu-linux_x86_64.whl
* Python 3.4: https://cntk.ai/PythonWheel/GPU/cntk-2.3-Pre-cp34-cp34m-linux_x86_64.whl
* Python 3.5: https://cntk.ai/PythonWheel/GPU/cntk-2.3-Pre-cp35-cp35m-linux_x86_64.whl
* Python 3.6: https://cntk.ai/PythonWheel/GPU/cntk-2.3-Pre-cp36-cp36m-linux_x86_64.whl


See more in the [Release Notes](https://docs.microsoft.com/en-us/cognitive-toolkit/ReleaseNotes/CNTK_2_2_Release_Notes).  
Get the Release from the [CNTK Releases page](https://github.com/Microsoft/CNTK/releases).

See [all news](https://docs.microsoft.com/en-us/cognitive-toolkit/news)

## Introduction

The Microsoft Cognitive Toolkit (https://cntk.ai), is a unified deep-learning toolkit that describes neural networks as a series of computational steps via a directed graph. In this directed graph, leaf nodes represent input values or network parameters, while other nodes represent matrix operations upon their inputs. CNTK allows to easily realize and combine popular model types such as feed-forward DNNs, convolutional nets (CNNs), and recurrent networks (RNNs/LSTMs). It implements stochastic gradient descent (SGD, error backpropagation) learning with automatic differentiation and parallelization across multiple GPUs and servers. CNTK has been available under an open-source license since April 2015. It is our hope that the community will take advantage of CNTK to share ideas more quickly through the exchange of open source working code.

## Installation

* [Setup CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine)
    * Windows [Python-only](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-windows-python) / [Script-driven](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-windows-binary-script) / [Manual](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-windows-binary-manual)
    * Linux [Python-only](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-linux-python) / [Script-driven](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-linux-binary-script) / [Manual](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-linux-binary-manual) / [Docker](https://docs.microsoft.com/en-us/cognitive-toolkit/cntk-docker-containers)
* [CNTK backend for Keras](https://docs.microsoft.com/en-us/cognitive-toolkit/using-cntk-with-keras)
* [Setup CNTK development environment](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-development-environment)
    * Windows [Script-driven](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-with-script-on-windows) / [Manual](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-windows)
    * Linux [Manual](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-linux)

### Nightly packages
If you prefer to use latest CNTK bits from master, use one of the CNTK nightly package.
* [Nightly packages for Windows](https://cntk.ai/nightly-windows.html)
* [Nightly packages for Linux](https://cntk.ai/nightly-linux.html)

## Learning CNTK

You may learn more about CNTK with the following resources:
* [General documentation](https://docs.microsoft.com/en-us/cognitive-toolkit/)
* [Python API documentation](https://cntk.ai/pythondocs/)
* [BrainScript documentation](https://docs.microsoft.com/en-us/cognitive-toolkit/Using-CNTK-with-BrainScript)
* [Evaluation documentation (C++, C#/.NET, Python, Java)](https://docs.microsoft.com/en-us/cognitive-toolkit/CNTK-Evaluation-Overview)
* [Manual](https://github.com/Microsoft/CNTK/tree/master/Manual)
* [Tutorials](https://docs.microsoft.com/en-us/cognitive-toolkit/tutorials)
* [Examples](https://docs.microsoft.com/en-us/cognitive-toolkit/Examples)
* [Pretrained models](./PretrainedModels)
* [Blog](https://www.microsoft.com/en-us/cognitive-toolkit/blog/)
* [Presentations](https://docs.microsoft.com/en-us/cognitive-toolkit/Presentations)
* [License](./LICENSE.md)

## More information

* [Reasons to switch from TensorFlow to CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/reasons-to-switch-from-tensorflow-to-cntk)
* [Contribute to CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/Contributing-to-CNTK)
* [FAQ](https://docs.microsoft.com/en-us/cognitive-toolkit/CNTK-FAQ)
* [Feedback](https://docs.microsoft.com/en-us/cognitive-toolkit/Feedback-Channels)

## Disclaimer

CNTK is in active use at Microsoft and constantly evolving. There will be bugs.

## Microsoft Open Source Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
