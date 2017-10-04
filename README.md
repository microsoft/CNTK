[![Join the chat at https://gitter.im/Microsoft/CNTK](https://badges.gitter.im/Microsoft/CNTK.svg)](https://gitter.im/Microsoft/CNTK?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## Latest news 

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

***2017-08-04.*** CNTK August interation plan posted [here](https://github.com/Microsoft/CNTK/issues/2194). 

***2017-07-31.* CNTK 2.1**  
Release of Cognitive Toolkit v.2.1.

Highlights:
* cuDNN 6.0 integration
* Support of Universal Windows Platform (UWP)
* Improvements in backend for Keras
* Performance improvements
* New manuals, tutorials and examples
* Multiple bug fixes

See more in the [Release Notes](https://docs.microsoft.com/en-us/cognitive-toolkit/ReleaseNotes/CNTK_2_1_Release_Notes).  
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
