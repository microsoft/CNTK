# Aug-Sept 2017 Notes

## Documentation

### Add HTML version of tutorials and manuals so that they can be searchable
### Add missing evaluation documents

## System 

### 16bit support for training on Volta GPU (limited functionality)
### Update learner interface to simplify parameter setting and adding new learners (**Potential breaking change**) 
### A preliminary C#/.NET API that enables people to train simple networks such as ConvNet on MNIST. 
### R-binding for training and evaluation (will be published in a separate repository) 
### Improve statistics for distributed evaluation 

## Examples
### Faster R-CNN object detection 
### New example for natural language processing (NLP) 
### Semantic segmentation (stretch goal) 

## Operations
### Noise contrastive estimation node 
### Aggregation on sparse gradient for embedded layer
### Gradient as an operator (stretch goal) 
### Reduced rank for convolution in C++ to enable convolution on 1D data 
### Dilated convolution 
Add support to dilation convolution on the GPU, exposed by BrainScript, C++ and Python API. Dilation convolution effectively increase the kernel size, without actually requiring a big kernel. To use dilation convoluton you need at least cuDNN 6.0. 
### Deterministic Pooling
Now call `cntk.debug.force_deterministic()` will make max and average pooling determistic, this behavior depend on cuDNN version 6 or later.

## Performance 
### Asynchronous evaluation API (Python and C#) 
### Intel MKL update to improve inference speed on CPU by around 2x on AlexNet 

## Keras and Tensorboard 
### Example on Keras and SKLearn multi-GPU support on CNTK 
### Image feature support with Tensorboard for CNTK 

## Others 
### Continue work on [Deep Learning Explained](https://www.edx.org/course/deep-learning-explained-microsoft-dat236x) course on edX. 
