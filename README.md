# CNTK

## Latest news
*2016-07-15.* V 1.6 Binary release  
CNTK v.1.6 binaries are on the [CNTK Releases page](https://github.com/Microsoft/CNTK/releases)

*2016-07-12.* We have further expanded Licensing options for CNTK 1bit-SGD and related components. See the details at the [Wiki page](https://github.com/microsoft/cntk/wiki/CNTK-1bit-SGD-License). These new options are an extension of the new CNTK 1bit-SGD License that we have announced on Jun 23, 2016.

*2016-07-05.* CNTK now supports *Deconvolution* and *Unpooling*. See the usage example in the Network number 4 in [MNIST Sample](https://github.com/Microsoft/CNTK/blob/master/Examples/Image/MNIST/README.md).

*2016-06-23.* New License Terms for CNTK 1bit-SGD and related components.  
Effective immediately the License Terms for CNTK 1bit-SGD and related components have changed. The new Terms provide more flexibility and enable new usage scenarios, especially in commercial environments. Read the new Terms at the [standard location](https://cntk1bitsgd.codeplex.com/license). Please note, that while the new Terms are significantly more flexible comparing to the previous ones, they are still **more restrictive** than the main CNTK License. Consequently everything described in [Enabling 1bit-SGD](https://github.com/Microsoft/CNTK/wiki/Enabling-1bit-SGD) section of the Wiki remains valid.

*2016-06-20.* A [post](http://itpeernetwork.intel.com/accelerating-the-computational-network-tool-kit-with-intel-mkl/) on Intel MKL and CNTK is published in the [Intel IT Peer Network](http://itpeernetwork.intel.com/accelerating-the-computational-network-tool-kit-with-intel-mkl/)

See [all news](https://github.com/Microsoft/CNTK/wiki/News).

## What is CNTK
CNTK (http://www.cntk.ai/), the Computational Network Toolkit by Microsoft Research, is a unified deep-learning toolkit that describes neural networks as a series of computational steps via a directed graph. In this directed graph, leaf nodes represent input values or network parameters, while other nodes represent matrix operations upon their inputs. CNTK allows to easily realize and combine popular model types such as feed-forward DNNs, convolutional nets (CNNs), and recurrent networks (RNNs/LSTMs). It implements stochastic gradient descent (SGD, error backpropagation) learning with automatic differentiation and parallelization across multiple GPUs and servers. CNTK has been available under an open-source license since April 2015. It is our hope that the community will take advantage of CNTK to share ideas more quickly through the exchange of open source working code.

Wiki: Go to the [CNTK Wiki](https://github.com/Microsoft/CNTK/wiki) for all information on CNTK including [setup](https://github.com/Microsoft/CNTK/wiki/Setup-CNTK-on-your-machine ), [examples](https://github.com/Microsoft/CNTK/wiki/Examples ), etc.

License: See [LICENSE.md](./LICENSE.md) in the root of this repository for the full license information.

Tutorial: [Microsoft Computational Network Toolkit (CNTK) @ NIPS 2015 Workshops](http://research.microsoft.com/en-us/um/people/dongyu/CNTK-Tutorial-NIPS2015.pdf)

Blogs:  

* [Microsoft Computational Network Toolkit offers most efficient distributed deep learning computational performance](http://blogs.technet.com/b/inside_microsoft_research/archive/2015/12/07/microsoft-computational-network-toolkit-offers-most-efficient-distributed-deep-learning-computational-performance.aspx)
* [Microsoft researchers win ImageNet computer vision challenge (December 2015)](http://blogs.microsoft.com/next/2015/12/10/microsoft-researchers-win-imagenet-computer-vision-challenge/)

## Performance

The figure below compares processing speed (frames processed per second) of CNTK to that of four other well-known toolkits. The configuration uses a fully connected 4-layer neural network (see our benchmark [scripts](https://github.com/Alexey-Kamenev/Benchmarks)) and an effective mini batch size (8192). All results were obtained on the same hardware with the respective latest public software versions as of Dec 3, 2015.

![Performance chart](Documentation/Documents/PerformanceChart.png)

## Citation

If you used this toolkit or part of it to do your research, please cite the work as:

Amit Agarwal, Eldar Akchurin, Chris Basoglu, Guoguo Chen, Scott Cyphers, Jasha Droppo, Adam Eversole, Brian Guenter, Mark Hillebrand, T. Ryan Hoens, Xuedong Huang, Zhiheng Huang, Vladimir Ivanov, Alexey Kamenev, Philipp Kranen, Oleksii Kuchaiev, Wolfgang Manousek, Avner May, Bhaskar Mitra, Olivier Nano, Gaizka Navarro, Alexey Orlov, Hari Parthasarathi, Baolin Peng, Marko Radmilac, Alexey Reznichenko, Frank Seide, Michael L. Seltzer, Malcolm Slaney, Andreas Stolcke, Huaming Wang, Yongqiang Wang, Kaisheng Yao, Dong Yu, Yu Zhang, Geoffrey Zweig (in alphabetical order), ["An Introduction to Computational Networks and the Computational Network Toolkit"](http://research.microsoft.com/apps/pubs/?id=226641), Microsoft Technical Report MSR-TR-2014-112, 2014.

## Disclaimer 

CNTK is in active use at Microsoft and constantly evolving. There will be bugs.


## Microsoft Open Source Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.