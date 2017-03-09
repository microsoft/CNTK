**The [CNTK Wiki](https://github.com/Microsoft/CNTK/wiki) has all information on CNTK including [setup](https://github.com/Microsoft/CNTK/wiki/Setup-CNTK-on-your-machine), [examples](https://github.com/Microsoft/CNTK/wiki/Examples), etc.**

Effective January 25, 2017 CNTK [1-bit Stochastic Gradient Descent (1bit-SGD)](https://github.com/Microsoft/CNTK/wiki/Enabling-1bit-SGD) and [BlockMomentumSGD](https://github.com/Microsoft/CNTK/wiki/Multiple-GPUs-and-machines#22-block-momentum-sgd) code is moved to a new Repository in GitHub. Read [this article](https://github.com/Microsoft/CNTK/wiki/Update-1bit-SGD-Submodule-Location) for details.

Give us feedback through these [channels](https://github.com/Microsoft/CNTK/wiki/Feedback-Channels).

# Latest news
***2017-02-28.* V 2.0 Beta 12 Release available at Docker Hub**  
CNTK V 2.0 Beta 12 Runtime packages are now available as [Public Images at Docker Hub](https://hub.docker.com/r/microsoft/cntk/).  
See more on CNTK as Docker Images in this [Wiki article](https://github.com/Microsoft/CNTK/wiki/CNTK-Docker-Containers).

***2017-02-23.* V 2.0 Beta 12 Release**  
Highlights of this Release:
* New and updated features: new activation functions, support of `Argmax` and `Argmin`, improved performance of `numpy` interop, new functionality of existing operators, and more.
* [CNTK for CPU on Windows can now be installed via `pip install` on Anaconda 3](https://github.com/Microsoft/CNTK/wiki/Setup-CNTK-on-your-machine). Other configurations will be enabled soon.
* HTK deserializers are now exposed in Python. All deserializers are exposed in C++.
* The memory pool implementation of CNTK has been updated with a new global optimization algorithm. Hyper memory compression has been removed.
* New features in C++ API.
* [New Eval examples for RNN models](https://github.com/Microsoft/CNTK/blob/master/Examples/Evaluation/CNTKLibraryCSEvalCPUOnlyExamples/CNTKLibraryCSEvalExamples.cs).
* New [CNTK NuGet Packages](https://github.com/Microsoft/CNTK/wiki/NuGet-Package) with CNTK V2 C++ Library.

See more in the [Release Notes](https://github.com/Microsoft/CNTK/wiki/CNTK_2_0_beta_12_Release_Notes).  
Get the Release from the [CNTK Releases page](https://github.com/Microsoft/CNTK/releases).

***2017-02-13.* V 2.0 Beta 11 Release available at Docker Hub**  
CNTK V 2.0 Beta 11 Runtime packages are now available as [Public Images at Docker Hub](https://hub.docker.com/r/microsoft/cntk/).  
See more on CNTK as Docker Images in this [Wiki article](https://github.com/Microsoft/CNTK/wiki/CNTK-Docker-Containers).

***2017-02-10.* V 2.0 Beta 11 Release**  
Highlights of this Release:
* New and updated core and Python API features.
* New Tutorials and Examples:
  * [Transfer learning example to train custom image classifiers](https://github.com/Microsoft/CNTK/tree/v2.0.beta11.0/Examples/Image/TransferLearning).
* New [CNTK NuGet Packages](https://github.com/Microsoft/CNTK/wiki/NuGet-Package).
  * Note a *breaking change* due to Assembly Strong Name enabling. See [Release Notes](https://github.com/Microsoft/CNTK/wiki/CNTK_2_0_beta_11_Release_Notes).

See more in the [Release Notes](https://github.com/Microsoft/CNTK/wiki/CNTK_2_0_beta_11_Release_Notes).  
Get the Release from the [CNTK Releases page](https://github.com/Microsoft/CNTK/releases).

***2017-02-08.* V 2.0 Beta 10 Release available at Docker Hub**  
CNTK V 2.0 Beta 10 Runtime packages are now available as [Public Images at Docker Hub](https://hub.docker.com/r/microsoft/cntk/).  
See more on CNTK as Docker Images in this [Wiki article](https://github.com/Microsoft/CNTK/wiki/CNTK-Docker-Containers).

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

Cognitive Toolkit (CNTK) provides significant performance gains compared to other toolkits [click here for details](https://arxiv.org/pdf/1608.07249.pdf). Here is a summary of findings by researchers at HKBU.

> * CNTK’s LSTM performance is 5-10x faster than the other toolkits.
> * For convolution (image tasks), CNTK is comparable, but note the authors were using CNTK 1.7.2, and current CNTK 2.0 beta 10 is over 30% faster than 1.7.2.
> * For all networks, CTNK's performance was superior to TensorFlow performance.

Historically, CNTK has been a pioneer in optimizing performance on multi-GPU systems. We continue to maintain the edge ([NVidia news at SuperComputing 2016](http://nvidianews.nvidia.com/news/nvidia-and-microsoft-accelerate-ai-together) and [CRAY at NIPS 2016](https://www.onmsft.com/news/microsoft-and-cray-announce-partnership-to-speed-up-deep-learning-on-supercomputers)).

 CNTK was a pioneer in introducing scalability across multi-server multi-GPU systems. The figure below compares processing speed (frames processed per second) of CNTK to that of four other well-known toolkits. The configuration uses a fully connected 4-layer neural network (see our benchmark [scripts](https://github.com/Alexey-Kamenev/Benchmarks)) and an effective mini batch size (8192). All results were obtained on the same hardware with the respective latest public software versions as of Dec 3, 2015.

![Performance chart](Documentation/Documents/PerformanceChart.png)

## Citation

If you used this toolkit or part of it to do your research, please cite the work as:

Amit Agarwal, Eldar Akchurin, Chris Basoglu, Guoguo Chen, Scott Cyphers, Jasha Droppo, Adam Eversole, Brian Guenter, Mark Hillebrand, T. Ryan Hoens, Xuedong Huang, Zhiheng Huang, Vladimir Ivanov, Alexey Kamenev, Philipp Kranen, Oleksii Kuchaiev, Wolfgang Manousek, Avner May, Bhaskar Mitra, Olivier Nano, Gaizka Navarro, Alexey Orlov, Hari Parthasarathi, Baolin Peng, Marko Radmilac, Alexey Reznichenko, Frank Seide, Michael L. Seltzer, Malcolm Slaney, Andreas Stolcke, Huaming Wang, Yongqiang Wang, Kaisheng Yao, Dong Yu, Yu Zhang, Geoffrey Zweig (in alphabetical order), ["An Introduction to Computational Networks and the Computational Network Toolkit"](https://research.microsoft.com/apps/pubs/?id=226641), Microsoft Technical Report MSR-TR-2014-112, 2014.

## Disclaimer

CNTK is in active use at Microsoft and constantly evolving. There will be bugs.

## Microsoft Open Source Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
