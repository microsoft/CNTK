**The [CNTK Wiki](https://github.com/Microsoft/CNTK/wiki) has all information on CNTK including [setup](https://github.com/Microsoft/CNTK/wiki/Setup-CNTK-on-your-machine), [examples](https://github.com/Microsoft/CNTK/wiki/Examples), etc.**

Give us feedback through these [channels](https://github.com/Microsoft/CNTK/wiki/Feedback-Channels).

# Latest news
***2016-01-19.* V 2.0 Beta 8 Release available at Docker Hub**  
CNTK V 2.0 Beta 8 Runtime packages are now available as [Public Images at Docker Hub](https://hub.docker.com/r/microsoft/cntk/).  
See more on CNTK as Docker Images in this [Wiki article](https://github.com/Microsoft/CNTK/wiki/CNTK-Docker-Containers).

***2016-01-16.* V 2.0 Beta 8 Release**  
Highlights of this Release:
* Support of Python v. 2.7, 3.4, and 3.5. See [binary and source setup](https://github.com/Microsoft/CNTK/wiki/Setup-CNTK-on-your-machine) instructions to find out about how to select Python version.
* New Python API features.
* New Python example [Feature extraction using a trained model in Python API](https://github.com/Microsoft/CNTK/tree/v2.0.beta8.0/Examples/Image/FeatureExtraction).
* Support of [Visual Studio 2015](https://github.com/Microsoft/CNTK/wiki/Setup-Migrate-VS13-to-VS15) for Windows version.
* Introduction of [C# API in CNTK Evaluation Library](https://github.com/Microsoft/CNTK/wiki/CNTK-Library-Managed-API) and a new set of [CNTK NuGet Packages](https://github.com/Microsoft/CNTK/wiki/NuGet-Package).
* CNTK Runtime packages are now available as [Public Images at Docker Hub](https://github.com/Microsoft/CNTK/wiki/CNTK-Docker-Containers). (**Beta 7** is currently available; Beta 8 Images availability will be announced separately in a few days)
* Version 3 of [CNTK Custom MKL Library](https://cntk.ai/mkl/) is available.

See more in the [Release Notes](https://github.com/Microsoft/CNTK/wiki/CNTK_2_0_beta_8_Release_Notes).  
Get the Release from the [CNTK Releases page](https://github.com/Microsoft/CNTK/releases).

***2017-01-10.* CNTK for Windows supports Visual 2015**

If you pull or merge the master branch, CNTK will now require Visual Studio 2015 to build on Windows. There are two ways to move your development environment to Visual Studio 2015:

* [Migrate VS2013 to VS2015](https://github.com/Microsoft/CNTK/wiki/Setup-Migrate-VS13-to-VS15): This gives you a fine grained control over where components are installed 
* [Script driven setup](https://github.com/Microsoft/CNTK/wiki/Setup-CNTK-with-script-on-Windows): This gives you an mostly automated migration to Visual Studio 2015

***2016-12-22.* V 2.0 Beta 7 Release**
Highlights of this Release:

* Python API behaviour is changed to be more strict.
* New Examples and Tutorials ([Artistic Style Transfer](https://github.com/Microsoft/CNTK/blob/v2.0.beta7.0/Tutorials/CNTK_205_Artistic_Style_Transfer.ipynb)
and [GoogLeNet (Inception V3)](https://github.com/Microsoft/CNTK/tree/v2.0.beta7.0/Examples/Image/Classification/GoogLeNet)
).
* New version of CNTK Evaluation library NuGet Package.
* CNTK Background Improvements.

See more in the [Release Notes](https://github.com/Microsoft/CNTK/wiki/CNTK_2_0_beta_7_Release_Notes)
Get the Release from the [CNTK Releases page](https://github.com/Microsoft/CNTK/releases)

***2016-12-13.* V 2.0 Beta 6 Release**
Highlights of this Release:

* Both Windows and Linux packages are now created using NVIDIA CUDA 8.0 toolkit.
* Linux version now supports Python 3.5 (Windows support is coming soon).
* Support for training on one-hot and sparse arrays via NumPy.
* New Examples and Tutorials: [Video action recognition](https://github.com/Microsoft/CNTK/tree/v2.0.beta6.0/Examples/Video/GettingStarted), [Finance Timeseries with Pandas/Numpy](https://github.com/Microsoft/CNTK/blob/v2.0.beta6.0/Tutorials/CNTK_104_Finance_Timeseries_Basic_with_Pandas_Numpy.ipynb), [Neural Character Language Models](https://github.com/Microsoft/CNTK/tree/v2.0.beta6.0/Examples/Text/CharacterLM/README.md)
* Stability Improvements and bug fixes.

See more in the [Release Notes](https://github.com/Microsoft/CNTK/wiki/CNTK_2_0_beta_6_Release_Notes)
Get the Release from the [CNTK Releases page](https://github.com/Microsoft/CNTK/releases)

See [all news](https://github.com/Microsoft/CNTK/wiki/News).

# What is The Microsoft Cognitive Toolkit

The Microsoft Cognitive Toolkit (https://www.microsoft.com/en-us/research/product/cognitive-toolkit/), is a unified deep-learning toolkit that describes neural networks as a series of computational steps via a directed graph. In this directed graph, leaf nodes represent input values or network parameters, while other nodes represent matrix operations upon their inputs. CNTK allows to easily realize and combine popular model types such as feed-forward DNNs, convolutional nets (CNNs), and recurrent networks (RNNs/LSTMs). It implements stochastic gradient descent (SGD, error backpropagation) learning with automatic differentiation and parallelization across multiple GPUs and servers. CNTK has been available under an open-source license since April 2015. It is our hope that the community will take advantage of CNTK to share ideas more quickly through the exchange of open source working code.

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
