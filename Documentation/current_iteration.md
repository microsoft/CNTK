# Aug-Sept 2017 Notes

## Breaking change
### This iteration requires cuDNN 6.0 in order to support dilated convolution and deterministic pooling. Please update your cuDNN.

## Documentation

### Add HTML version of tutorials and manuals so that they can be searchable
We have added HTML versions of the tutorials and manuals with the Python documentation. This makes the [tutorial notebooks](https://www.cntk.ai/pythondocs/tutorials.html) and manuals searchable as well.

### Add missing evaluation documents

## System 

### 16bit support for training on Volta GPU (limited functionality)
### Update learner interface to simplify parameter setting and adding new learners (**Potential breaking change**) 
### A C#/.NET API that enables people to build and train networks. 
##### Basic training support is added to C#/.NET API. New training examples include:
##### 1. Convolution neural network for image classification of the MNIST dataset. (https://github.com/Microsoft/CNTK/tree/master/Examples/TrainingCShape/Common/MNISTClassifier.cs)
##### 2. Build, train, and evaluate a ResNet model with C#/.NET API. (https://github.com/Microsoft/CNTK/tree/master/Examples/TrainingCShape/Common/CifarResNetClassifier.cs)
##### 3. Transfer learning with C#/.NET API. (https://github.com/Microsoft/CNTK/tree/master/Examples/TrainingCShape/Common/TransferLearning.cs)
##### 4. Build and train a LSTM sequence classifier with C#/.NET API. (https://github.com/Microsoft/CNTK/tree/master/Examples/TrainingCShape/Common/LSTMSequenceClassifier.cs)

### R-binding for training and evaluation (will be published in a separate repository) 
### Improve statistics for distributed evaluation 

## Examples
### Faster R-CNN object detection 
### Support for bounding box regression and VGG model in Fast R-CNN
### New tutorial on Faster R-CNN object detection and updated tutorial on Fast R-CNN
### Object detection demo script that allows to choose different detectors, base models and data sets
### New example for natural language processing (NLP) 
### Semantic segmentation (stretch goal) 
### New C++ Eval Examples
The C++ examples [`CNTKLibraryCPPEvalCPUOnlyExamples`](https://github.com/Microsoft/CNTK/tree/release/2.2/Examples/Evaluation/CNTKLibraryCPPEvalCPUOnlyExamples) and [`CNTKLibraryCPPEvalGPUExamples`](https://github.com/Microsoft/CNTK/tree/release/2.2/Examples/Evaluation/CNTKLibraryCPPEvalGPUExamples) illustrate how to use C++ CNTK Library for model evaluation on CPU and GPU. The [UWPImageRecognition](https://github.com/Microsoft/CNTK/tree/release/2.2/Examples/Evaluation/UWPImageRecognition) contains an example using CNTK UWP library for model evaluation.
### Add new C# Eval examples
  * asynchronous evaluation:  [`EvaluationSingleImageAsync()`](https://github.com/Microsoft/CNTK/tree/release/2.2/Examples/Evaluation/CNTKLibraryCSEvalCPUOnlyExamples/CNTKLibraryCSEvalExamples.cs),
  * evaluating intermediate layers: [`EvaluateIntermediateLayer()`](https://github.com/Microsoft/CNTK/tree/release/2.2/Examples/Evaluation/CNTKLibraryCSEvalCPUOnlyExamples/CNTKLibraryCSEvalExamples.cs),
  * evaluating outputs from multiple nodes: [`EvaluateCombinedOutputs()`](https://github.com/Microsoft/CNTK/tree/release/2.2/Examples/Evaluation/CNTKLibraryCSEvalCPUOnlyExamples/CNTKLibraryCSEvalExamples.cs).

## Operations
### Noise contrastive estimation node

This provides a built-in efficient (but approximate) loss function used to train networks when the 
number of classes is very large. For example you can use it when you want to predict the next word 
out of a vocabulary of tens or hundreds of thousands of words.

To use it define your loss as 
```python
loss = nce_loss(weights, biases, inputs, labels, noise_distribution)
```
and once you are done training you can make predictions like this
```python
logits = C.times(weights, C.reshape(inputs, (1,), 1)) + biases
```
Note that the noise contrastive estimation loss cannot help with 
reducing inference costs; the cost savings are only during training.

### Improved AttentionModel

A bug in our AttentionModel layer has been fixed and we now faithfully implement the paper

> Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et. al.)

Furthermore, the arguments `attention_span` and `attention_axis` of the AttentionModel
have been **deprecated**. They should be left to their default values, in which case the attention is computed over the whole sequence
and the output is a sequence of vectors of the same dimension as the first argument over the axis of the second argument.
This also leads to substantial speed gains (our CNTK 204 Tutorial now runs more than 2x faster). 

### Aggregation on sparse gradient for embedded layer
#### This change saves costly conversion from sparse to dense before gradient aggregation when embedding vocabulary size is huge.
#### It is currently enabled for GPU build when training on GPU with non-quantized data parallel SGD. For other distributed learners and CPU build, it is disabled by default.
#### It can be manually turned off in python by calling `cntk.cntk_py.use_sparse_gradient_aggregation_in_data_parallel_sgd(False)`
#### Note that for a rare case of running distributed training with CPU device on a GPU build, you need to manually turn it off to avoid unimplemented exception
### Gradient as an operator (stretch goal) 
### Reduced rank for convolution in C++ to enable convolution on 1D data 
Now convolution and convolution_transpose support data without channel or depth dimension by setting reductionRank to 0 instead of 1.
### Dilated convolution 
Add support to dilation convolution on the GPU, exposed by BrainScript, C++ and Python API. Dilation convolution effectively increase the kernel size, without actually requiring a big kernel. To use dilation convoluton you need at least cuDNN 6.0. 
### Deterministic Pooling
Now call `cntk.debug.force_deterministic()` will make max and average pooling determistic, this behavior depend on cuDNN version 6 or later.

## Performance 
### Asynchronous evaluation API (Python and C#) 
### Intel MKL update to improve inference speed on CPU by around 2x on AlexNet 

## Keras and Tensorboard 
### Example on Keras and SKLearn multi-GPU support on CNTK 
### Added Tensorboard image support for CNTK. Now CNTK users can use tensorboard to display images. More details and examples can be found [here](http://docs.microsoft.com/en-us/cognitive-toolkit/Using-TensorBoard-for-Visualization).

## Others 
### Continue work on [Deep Learning Explained](https://www.edx.org/course/deep-learning-explained-microsoft-dat236x) course on edX. 
