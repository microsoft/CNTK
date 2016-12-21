**The [CNTK Wiki](https://github.com/Microsoft/CNTK/wiki) has all information on CNTK including [setup](https://github.com/Microsoft/CNTK/wiki/Setup-CNTK-on-your-machine ), [examples](https://github.com/Microsoft/CNTK/wiki/Examples ), etc.**

# Latest news
***2016-12-13.*** V 2.0 Beta 6 Release  
Highlights of this Release:
* Both Windows and Linux packages are now created using NVIDIA CUDA 8.0 toolkit.
* Linux version now supports Python 3.5 (Windows support is coming soon).
* Support for training on one-hot and sparse arrays via NumPy.
* New Examples and Tutorials: [Video action recognition](https://github.com/Microsoft/CNTK/tree/v2.0.beta6.0/Examples/Video/GettingStarted), [Finance Timeseries with Pandas/Numpy](https://github.com/Microsoft/CNTK/blob/v2.0.beta6.0/Tutorials/CNTK_104_Finance_Timeseries_Basic_with_Pandas_Numpy.ipynb), [Neural Character Language Models](https://github.com/Microsoft/CNTK/tree/v2.0.beta6.0/Examples/Text/CharacterLM/README.md)
* Stability Improvements and bug fixes.

See more in the [Release Notes](https://github.com/Microsoft/CNTK/wiki/CNTK_2_0_beta_6_Release_Notes)  
Get the Release from the [CNTK Releases page](https://github.com/Microsoft/CNTK/releases)

***2016-11-25.*** V 2.0 Beta 5 Release  
Highlights of this Release:
* The Windows binary packages are now created using the NVIDIA CUDA 8 toolkit, see the [release notes](https://github.com/Microsoft/CNTK/wiki/CNTK_2_0_beta_5_Release_Notes) for details. The CNTK-Linux binary packages are still built with CUDA 7.5. The Linux support for Cuda8 will follow shortly!
* Performance enhancements for evaluation of bitmap images through the new `EvaluateRgbImage` function in the [managed Eval API](https://github.com/Microsoft/CNTK/wiki/Managed-EvalDLL-API).
* A new version of the [CNTK Nuget package](https://github.com/Microsoft/CNTK/wiki/NuGet-Package) is available. 
* Stability Improvements and bug fixes, i.e. decreased memory footprint in CNTK Text Format deserializer. 
* We continue to improve documentation and tutorials on an ongoing basis, in this release we added a [Sequence-to-Sequence tutorial](https://github.com/Microsoft/CNTK/blob/v2.0.beta5.0/Tutorials/CNTK_204_Sequence_To_Sequence.ipynb).

See more in the [Release Notes](https://github.com/Microsoft/CNTK/wiki/CNTK_2_0_beta_5_Release_Notes)  
Get the Release from the [CNTK Releases page](https://github.com/Microsoft/CNTK/releases)

***2016-11-21.*** V 2.0 Beta 4 Release  
Highlights of this Release:
* New ASGD/Hogwild! training using Microsoft’s Parameter Server ([Project Multiverso](https://github.com/Microsoft/multiverso))
* Distributed Scenarios now supported in CNTK Python API
* New [Memory Compression](https://github.com/Microsoft/CNTK/wiki/Top-level-configurations#hypercompressmemory) mode to reduce memory usage on GPU
* CNTK Docker image with 1bit-SGD support
* Stability Improvements and bug fixes

See more in the [Release Notes](https://github.com/Microsoft/CNTK/wiki/CNTK_2_0_beta_4_Release_Notes)  
Get the Release from the [CNTK Releases page](https://github.com/Microsoft/CNTK/releases)

***2016-11-11.*** V 2.0 Beta 3 Release  
Highlights of this Release:
* Integration with [NVIDIA NCCL](https://github.com/NVIDIA/nccl). Works with Linux when building CNTK from sources. See here [how to enable](https://github.com/Microsoft/CNTK/wiki/Setup-CNTK-on-Linux#optional-nccl)
* The first V.2.0 Prerelease Nuget Package for CNTK Evaluation library
* Stability Improvements and bug fixes

See more in the [Release Notes](https://github.com/Microsoft/CNTK/wiki/CNTK_2_0_beta_3_Release_Notes)  
Get the Release from the [CNTK Releases page](https://github.com/Microsoft/CNTK/releases)

***2016-11-03.*** V 2.0 Beta 2 Release  
Highlights of this Release:
* Feature tuning and bug fixing based on the feedback on Beta 1
* Changes in the Examples and Tutorials based on the same feedback
* New [Tutorial on Reinforcement Learning](https://github.com/Microsoft/CNTK/blob/v2.0.beta2.0/bindings/python/tutorials/CNTK_203_Reinforcement_Learning_Basics.ipynb)

See more in the [Release Notes](https://github.com/Microsoft/CNTK/wiki/CNTK_2_0_beta_2_Release_Notes)  
Get the Release from the [CNTK Releases page](https://github.com/Microsoft/CNTK/releases)

See [all news](https://github.com/Microsoft/CNTK/wiki/News).

# What is The Microsoft Cognitive Toolkit
The Microsoft Cognitive Toolkit (https://www.cntk.ai/), is a unified deep-learning toolkit that describes neural networks as a series of computational steps via a directed graph. In this directed graph, leaf nodes represent input values or network parameters, while other nodes represent matrix operations upon their inputs. CNTK allows to easily realize and combine popular model types such as feed-forward DNNs, convolutional nets (CNNs), and recurrent networks (RNNs/LSTMs). It implements stochastic gradient descent (SGD, error backpropagation) learning with automatic differentiation and parallelization across multiple GPUs and servers. CNTK has been available under an open-source license since April 2015. It is our hope that the community will take advantage of CNTK to share ideas more quickly through the exchange of open source working code.

Wiki: Go to the [CNTK Wiki](https://github.com/Microsoft/CNTK/wiki) for all information on CNTK including [setup](https://github.com/Microsoft/CNTK/wiki/Setup-CNTK-on-your-machine ), [examples](https://github.com/Microsoft/CNTK/wiki/Examples ), etc.

License: See [LICENSE.md](./LICENSE.md) in the root of this repository for the full license information.

Tutorial: [Microsoft Computational Network Toolkit (CNTK) @ NIPS 2015 Workshops](https://research.microsoft.com/en-us/um/people/dongyu/CNTK-Tutorial-NIPS2015.pdf)

Blogs:  

* [Microsoft Computational Network Toolkit offers most efficient distributed deep learning computational performance](https://blogs.technet.com/b/inside_microsoft_research/archive/2015/12/07/microsoft-computational-network-toolkit-offers-most-efficient-distributed-deep-learning-computational-performance.aspx)
* [Microsoft researchers win ImageNet computer vision challenge (December 2015)](https://blogs.microsoft.com/next/2015/12/10/microsoft-researchers-win-imagenet-computer-vision-challenge/)

## Performance

The figure below compares processing speed (frames processed per second) of CNTK to that of four other well-known toolkits. The configuration uses a fully connected 4-layer neural network (see our benchmark [scripts](https://github.com/Alexey-Kamenev/Benchmarks)) and an effective mini batch size (8192). All results were obtained on the same hardware with the respective latest public software versions as of Dec 3, 2015.

![Performance chart](Documentation/Documents/PerformanceChart.png)

## Citation

If you used this toolkit or part of it to do your research, please cite the work as:

Amit Agarwal, Eldar Akchurin, Chris Basoglu, Guoguo Chen, Scott Cyphers, Jasha Droppo, Adam Eversole, Brian Guenter, Mark Hillebrand, T. Ryan Hoens, Xuedong Huang, Zhiheng Huang, Vladimir Ivanov, Alexey Kamenev, Philipp Kranen, Oleksii Kuchaiev, Wolfgang Manousek, Avner May, Bhaskar Mitra, Olivier Nano, Gaizka Navarro, Alexey Orlov, Hari Parthasarathi, Baolin Peng, Marko Radmilac, Alexey Reznichenko, Frank Seide, Michael L. Seltzer, Malcolm Slaney, Andreas Stolcke, Huaming Wang, Yongqiang Wang, Kaisheng Yao, Dong Yu, Yu Zhang, Geoffrey Zweig (in alphabetical order), ["An Introduction to Computational Networks and the Computational Network Toolkit"](https://research.microsoft.com/apps/pubs/?id=226641), Microsoft Technical Report MSR-TR-2014-112, 2014.

## Disclaimer 

CNTK is in active use at Microsoft and constantly evolving. There will be bugs.


## Microsoft Open Source Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
