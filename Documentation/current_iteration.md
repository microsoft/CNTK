# CNTK v2.2 Release Notes

## Breaking change
* This iteration *requires* cuDNN 6.0 in order to support dilated convolution and deterministic pooling. Please update your cuDNN.
* This iteration *requires* OpenCV to support TensorBoard Image feature. Please install OpenCV before you install CNTK.

## Documentation

### Add HTML version of tutorials and manuals so that they can be searchable
We have added HTML versions of the [tutorials](https://www.cntk.ai/pythondocs/tutorials.html) and [manuals](https://www.cntk.ai/pythondocs/manuals.html) with the [Python documentation](https://www.cntk.ai/pythondocs/). This makes the [tutorial notebooks](https://www.cntk.ai/pythondocs/tutorials.html) and manuals searchable as well.

### Updated evaluation documents
Documents related to model evaluation have been updated. Please check the latest documents [here](https://docs.microsoft.com/en-us/cognitive-toolkit/CNTK-Evaluation-Overview).

## System

### 16bit support for training on Volta GPU (limited functionality)
This work is rolled over into next release due to dependency on test infrastructure updates.

### Support for NCCL 2
Now [NCCL](https://developer.nvidia.com/nccl) can be used across machines. User need to enable NCCL in build configure as [here](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-linux).
Note:
* After installed the downloaded NCCL 2 package, there are two packages:
```
/var/nccl-repo-2.0.4-ga/libnccl2_2.0.4-1+cuda8.0_amd64.deb
/var/nccl-repo-2.0.4-ga/libnccl-dev_2.0.4-1+cuda8.0_amd64.deb.
```
Install both of them for building CNTK with NCCL 2.
* Due to issues in system configuration, user might encounter failure during NCCL initialization. To get detailed information about the failure, please set environment variable `NCCL_DEBUG=INFO`.
* There are known issues in current release of NCCL 2 on system configured with InfiniBand devices running in mixed IB and IPoIB modes. To use IB mode devices only, please set environment variable `NCCL_IB_HCA=devices` running on IB mode, e.g.:
```
export NCCL_IB_HCA=mlx5_0,mlx5_2
```

### CNTK learner interface update

This update simplifies the learner APIs and deprecates the concepts of `unitType.minibatch` and `UnitType.sample`. The purpose is to make the API intuitive to specify the learner hyper-parameters while preserving the unique model update techniques in CNTK --- the mean gradients of every N samples contributes approximately the same to the model updates regardless of the actual data minibatch sizes. Detailed explanation can be found at the manual on [How to Use CNTK Learners](https://github.com/Microsoft/CNTK/blob/master/Manual/Manual_How_to_use_learners.ipynb).

In the new API, all supported learners, including [AdaDelta](https://cntk.ai/pythondocs/cntk.learners.html#cntk.learners.adadelta),
[AdaGrad](https://cntk.ai/pythondocs/cntk.learners.html#cntk.learners.adagrad),
 [FSAdaGrad](https://cntk.ai/pythondocs/cntk.learners.html#cntk.learners.fsadagrad),
[Adam](https://cntk.ai/pythondocs/cntk.learners.html#cntk.learners.adam),
[MomentumSGD](https://cntk.ai/pythondocs/cntk.learners.html#cntk.learners.momentum_sgd),
[Nesterov](https://cntk.ai/pythondocs/cntk.learners.html#cntk.learners.nesterov),
[RMSProp](https://cntk.ai/pythondocs/cntk.learners.html#cntk.learners.rmsprop), and
[SGD](https://cntk.ai/pythondocs/cntk.learners.html#cntk.learners.sgd), can now be specified by
```python
cntk.<cntk_supporting_learner>(parameters=model.parametes,
    lr=<float or list>,
    [momentum=<float or list>], [variance_momentum=<float or list>],
    minibatch_size=<None, int, or cntk.learners.IGNORE>,
    ...other learner parameters)
```

There are two major changes:  

* `lr`: the learning rate schedule can be specified as a float, a list of floats, or a list of pairs (float, int) (see parameter definition at  [learning_parameter_schedule](https://cntk.ai/pythondocs/cntk.learners.html?highlight=learning_rate_schedule#cntk.learners.learning_parameter_schedule)). The same specification applies to the momentum and variance_moment of learners,
 [FSAdaGrad](https://cntk.ai/pythondocs/cntk.learners.html#cntk.learners.fsadagrad),
[Adam](https://cntk.ai/pythondocs/cntk.learners.html#cntk.learners.adam),
[MomentumSGD](https://cntk.ai/pythondocs/cntk.learners.html#cntk.learners.momentum_sgd),
[Nesterov](https://cntk.ai/pythondocs/cntk.learners.html#cntk.learners.nesterov),  where such hyper-parameters are required.

* `minibatch_size`: a minibatch_size can be specified to guarantee that the mean gradient of every N (minibatch_size=N) samples contribute to the model updates with the same learning rate even if the actual minibatch size of the data is different from N. This is useful when the data minibatch size varies, especially in scenarios of training with variable length sequences, and/or uneven data partition in distributed training.
    * If we set `minibatch_size=cntk.learners.IGNORE`, then we recover the behavior in the literature: The mean gradient of the whole minibatch contributes to the model update with the same learning rate. The behavior of ignoring the data minibatch data size is the same as specifying a minibatch size for the learner when the data minibatch size equals to the specified minibatch size.

With the new API:
* To have model updates in the same manner as in the classic deep learning literature, we can specify the learner by setting `minibatch_size=cntk.learners.IGNORE` to ignore the minibatch size, e.g.
```python
sgd_learner_m = C.sgd(z.parameters, lr = 0.5, minibatch_size = C.learners.IGNORE)
```
Note:
- To enable CNTK specific techniques which apply the same learning rate to the mean gradient of every N samples regardless of the actual minibatch sizes, we can specify the learner by setting `minibatch_size=N`, e.g. setting `minibatch_size=2`,
```python
sgd_learner_s2 = C.sgd(z.parameters, lr = 0.5, minibatch_size = 2)
```

Regarding the [momentum_schedule](https://cntk.ai/pythondocs/cntk.learners.html?highlight=learning_rate_schedule#cntk.learners.momentum_schedule) of the learners [FSAdaGrad](https://cntk.ai/pythondocs/cntk.learners.html#cntk.learners.fsadagrad),
[Adam](https://cntk.ai/pythondocs/cntk.learners.html#cntk.learners.adam),
[MomentumSGD](https://cntk.ai/pythondocs/cntk.learners.html#cntk.learners.momentum_sgd),
and [Nesterov](https://cntk.ai/pythondocs/cntk.learners.html#cntk.learners.nesterov), it can be specified in a similar way. Let's use `momentum_sgd` as an example:
```python
momentum_sgd(parameters, lr=float or list of floats, momentum=float or list of floats,
             minibatch_size=C.learners.IGNORE, epoch_size=epoch_size)
momentum_sgd(parameters, lr=float or list of floats, momentum=float or list of floats,
             minibatch_size=N, epoch_size=epoch_size)
```

Similar to `learning_rate_schedule`, the arguments are interpreted in the same way:

* With `minibatch_size=C.learners.IGNORE`, the decay `momentum=beta` is applied to the mean gradient of the whole minibatch regardless of its size. For example, regardless of the minibatch size either be N or 2N (or any size), the mean gradient of such a minibatch will have same decay factor beta.

* With `minibatch_size=N`, the decay `momentum=beta` is applied to the mean gradient of every N samples. For example,  minibatches of sizes N, 2N, 3N and kN will have decays of beta, pow(beta, 2), pow(beta, 3) and pow(beta, k) respectively --- the decay is exponential in the proportion of the actual minibatch size to the specified minibatch size.


### A C#/.NET API that enables people to build and train networks.

##### Training Support Is Added To C#/.NET API.
With this addition to the existing CNTK C# Evaluation API, .NET developers can enjoy fully a integrated deep learning experience. A deep neural network can be built, trained, and validated fully in C# while still taking advantage of CNTK performance strength. Users may debug directly into CNTK source code to see how a DNN is trained and evaluated. New features include:

##### Basic C# Training API.
Over 100 basic functions are supported to build a computation network. These functions include Sigmoid, Tanh, ReLU, Plus, Minus, Convolution, Pooling, BatchNormalization, to name a few.

As an example, to build a logistic regression loss function:
```cs
Function z = CNTKLib.Times(weightParam, input) + biasParam;
Function loss = CNTKLib.CrossEntropyWithSoftmax(z, labelVariable);
```

##### CNTK Function As A Primitive Element To Build A DNN
A DNN is built through basic operation composition. For example, to build a ResNet node:
```cs
Function conv = CNTKLib.Pooling(CNTKLib.Convolution(convParam, input),
                                PoolingType.Average, poolingWindowShape);
Function resNetNode = CNTKLib.ReLU(CNTKLib.Plus(conv, input));
```

##### Batching Support
We provide MinibatchSource and MinibatchData utilities to help efficient data loading and batching.

##### Training Support
We support many Stochastic Gradient Descent optimizers commonly seen in the DNN literature: MomentumSGDLearner, AdamLearner, AdaGradLearner, etc. For example, to train a model with a ADAM Stochastic Optimizer:
```cs
var parameterLearners = new List<Learner>() { Learner.AdamLearner(classifierOutput.Parameters(),
                                                                  learningRate, momentum) };
var trainer = Trainer.CreateTrainer(classifierOutput, trainingLoss,
                                    prediction, parameterLearners);
```

Training examples cover a broad range of DNN use cases:
* A hello-world example to train and evaluate a logistic regression model using C#/API:
https://github.com/Microsoft/CNTK/tree/master/Examples/TrainingCSharp/Common/LogisticRegression.cs
* Convolution neural network for image classification of the MNIST dataset:
https://github.com/Microsoft/CNTK/tree/master/Examples/TrainingCSharp/Common/MNISTClassifier.cs
* Build, train, and evaluate a ResNet model with C#/.NET API:  
https://github.com/Microsoft/CNTK/tree/master/Examples/TrainingCSharp/Common/CifarResNetClassifier.cs
* Transfer learning with C#/.NET API:  
https://github.com/Microsoft/CNTK/tree/master/Examples/TrainingCSharp/Common/TransferLearning.cs
* Build and train a LSTM sequence classifier with C#/.NET API:
https://github.com/Microsoft/CNTK/tree/master/Examples/TrainingCSharp/Common/LSTMSequenceClassifier.cs

### R-binding for CNTK
R-binding for CNTK, which enables both training and evaluation, will be published in a separate repository very soon.

## Examples
### Object Detection with Fast R-CNN and Faster R-CNN
* Support for bounding box regression and VGG model in Fast R-CNN.
* New tutorial in documentation on [Faster R-CNN object detection](https://docs.microsoft.com/en-us/cognitive-toolkit/Object-Detection-using-Faster-R-CNN) and updated tutorial on [Fast R-CNN](https://docs.microsoft.com/en-us/cognitive-toolkit/Object-Detection-using-Fast-R-CNN).
* [Object detection demo script](https://github.com/Microsoft/CNTK/tree/master/Examples/Image/Detection) that allows to choose different detectors, base models, and data sets.

### New C++ Eval Examples
We added new C++ examples [`CNTKLibraryCPPEvalCPUOnlyExamples`](https://github.com/Microsoft/CNTK/tree/release/2.2/Examples/Evaluation/CNTKLibraryCPPEvalCPUOnlyExamples) and [`CNTKLibraryCPPEvalGPUExamples`](https://github.com/Microsoft/CNTK/tree/release/2.2/Examples/Evaluation/CNTKLibraryCPPEvalGPUExamples). They illustrate how to use C++ CNTK Library for model evaluation on CPU and GPU. Another new example is [UWPImageRecognition](https://github.com/Microsoft/CNTK/tree/release/2.2/Examples/Evaluation/UWPImageRecognition), which is an example using CNTK UWP library for model evaluation.

### New C# Eval examples
We added an example for asynchronous evaluation:  [`EvaluationSingleImageAsync()`](https://github.com/Microsoft/CNTK/tree/release/2.2/Examples/Evaluation/CNTKLibraryCSEvalCPUOnlyExamples/CNTKLibraryCSEvalExamples.cs). One thing we shall point out is CNTK C# API does not have an asynchronous method for Evaluate(), because the evaluation is a CPU-bound operation (Please refer to [this article](https://blogs.msdn.microsoft.com/pfxteam/2012/03/24/should-i-expose-asynchronous-wrappers-for-synchronous-methods/) for detailed explanation). However, it is desired to run evaluation asynchronously in some use cases, e.g. offloading for responsiveness, we show in the example [`EvaluationSingleImageAsync()`](https://github.com/Microsoft/CNTK/blob/release/2.2/Examples/Evaluation/CNTKLibraryCSEvalCPUOnlyExamples/CNTKLibraryCSEvalExamples.cs) how to achieve that by using the extension method [`EvaluateAsync()`](https://github.com/Microsoft/CNTK/blob/release/2.2/Examples/Evaluation/CNTKLibraryCSEvalCPUOnlyExamples/CNTKExtensions.cs). Please refer to the section *Run evaluation asynchronously* on the page [Using C#/.NET Managed API](https://docs.microsoft.com/en-us/cognitive-toolkit/CNTK-Library-Evaluation-on-Windows#using-cnet-managed-api) for details.
* Evaluating intermediate layers: [`EvaluateIntermediateLayer()`](https://github.com/Microsoft/CNTK/tree/release/2.2/Examples/Evaluation/CNTKLibraryCSEvalCPUOnlyExamples/CNTKLibraryCSEvalExamples.cs)
* Evaluating outputs from multiple nodes: [`EvaluateCombinedOutputs()`](https://github.com/Microsoft/CNTK/tree/release/2.2/Examples/Evaluation/CNTKLibraryCSEvalCPUOnlyExamples/CNTKLibraryCSEvalExamples.cs)

## Operations
### Noise contrastive estimation node

This provides a built-in efficient (but approximate) loss function used to train networks when the
number of classes is very large. For example you can use it when you want to predict the next word
out of a vocabulary of tens or hundreds of thousands of words.

To use it define your loss as:
```python
loss = nce_loss(weights, biases, inputs, labels, noise_distribution)
```
and once you are done training you can make predictions like this
```python
logits = C.times(weights, C.reshape(inputs, (1,), 1)) + biases
```
Note that the noise contrastive estimation loss cannot help with reducing inference costs; the cost savings are only during training.

### Improved AttentionModel

A bug in our AttentionModel layer has been fixed and we now faithfully implement the paper

> Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et. al.)

Furthermore, the arguments `attention_span` and `attention_axis` of the AttentionModel
have been **deprecated**. They should be left to their default values, in which case the attention is computed over the whole sequence
and the output is a sequence of vectors of the same dimension as the first argument over the axis of the second argument.
This also leads to substantial speed gains (our CNTK 204 Tutorial now runs more than 2x faster).

### Aggregation on sparse gradient for embedded layer
This change saves costly conversion from sparse to dense before gradient aggregation when embedding vocabulary size is huge. It is currently enabled for GPU build when training on GPU with non-quantized data parallel SGD. For other distributed learners and CPU build, it is disabled by default. It can be manually turned off in python by calling `cntk.cntk_py.use_sparse_gradient_aggregation_in_data_parallel_sgd(False)`. Note that for a rare case of running distributed training with CPU device on a GPU build, you need to manually turn it off to avoid unimplemented exception

### Reduced rank for convolution in C++ to enable convolution on 1D data
Now `convolution` and `convolution_transpose` support data without channel or depth dimension by setting `reductionRank` to 0 instead of 1. The motivation for this change is to add the ability to natively support geometric data without the need to insert a dummy channel dimension through reshaping.

### Dilated convolution (GPU only)
We added support for dilation convolution on the GPU, exposed by BrainScript, C++ and Python API. Dilation convolution effectively increase the kernel size, without actually requiring a big kernel. To use dilation convolution you need at least cuDNN 6.0. Dilated convolution improved the result of image segmentation in https://arxiv.org/pdf/1511.07122.pdf, in addition it exponentially increase the receptive field without increasing the required memory. One thing to note is there is currently no implementation of dilated convolution on CPU, therefore you cannot evaluate a model containing dilated convolution on CPU.

### Free static axes support for convolution
* We have added support for free static axes `FreeDimension` for convolution. This allows changing the input tensor size from minibatch to minibatch. For example, in case of CNNs this allows each minibatch to potentially have a different underlying image size. Similar support has also been enabled for pooling node.
* Note that the Faster R-CNN example for object detection does not yet leverage the free static axes support for convolution (i.e., still scales and pads input images to a fixed size). This example is being updated to use free static axes for arbitrary input image sizes, and is targeted for next release.

### Deterministic Pooling
Now call `cntk.debug.force_deterministic()` will make max and average pooling deterministic, this behavior depend on cuDNN version 6 or later.

## Performance
### Intel MKL update to improve inference speed on CPU by around 2x on AlexNet
This work is rolled over to next release due to dependency on test infrastructure updates.

## Keras and Tensorboard
### Multi-GPU support for Keras on CNTK.

We added an article to elaborated how to conduct parallel training on CNTK with Keras. Details are [here](http://docs.microsoft.com/en-us/cognitive-toolkit/Using-CNTK-MultiGPU-Support-with-Keras).

### Tensorboard image support for CNTK.

We added the image feature support for TensorBoard. Now CNTK users can use TensorBoard to display images. More details and examples can be found [here](http://docs.microsoft.com/en-us/cognitive-toolkit/Using-TensorBoard-for-Visualization).

### Acknowledgments
We thank the following community members for their contributions:+

* [@cijianzy](https://github.com/cijianzy)
* [@duli2012](https://github.com/duli2012)
* [@FDecaYed](https://github.com/FDecaYed)
* [@karolzak](https://github.com/karolzak)
* [@KGWANGMIN](https://github.com/KGWANGMIN)
* [@ottolu](https://github.com/ottolu)
* [@StillKeepTry](https://github.com/StillKeepTry)
* [@taehoonlee](https://github.com/taehoonlee)

We apologize for any community contributions we might have overlooked in these release notes.

## Others
### Continue work on [Deep Learning Explained](https://www.edx.org/course/deep-learning-explained-microsoft-dat236x) course on edX.
