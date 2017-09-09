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

## Operations
### Noise contrastive estimation node 
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
