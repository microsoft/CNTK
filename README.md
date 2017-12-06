[![Join the chat at https://gitter.im/Microsoft/CNTK](https://badges.gitter.im/Microsoft/CNTK.svg)](https://gitter.im/Microsoft/CNTK?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## Latest news

***2017-12-05.* CNTK 2.3.1**
Release of Cognitive Toolkit v.2.3.1.

CNTK support for ONNX format is now out of preview mode.

If you want to try ONNX, you can build from master or `pip install` one of the below wheel that matches you Python environment.

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


***2017-09-25.*** CNTK September interation plan posted [here](https://github.com/Microsoft/CNTK/issues/2410).

***2017-09-24.*** CNTK R-binding now available [here](https://github.com/Microsoft/CNTK-R).

***2017-09-15.* CNTK 2.2**  
Release of Cognitive Toolkit v2.2.

Hightlights:
* NCCL 2 support
* New learner interface
* A C#/.NET API that enables people to build and train networks
* New C++ and C# eval examples
* New nodes
* Tensorboard image support for CNTK

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
