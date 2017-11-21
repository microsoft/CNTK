# CNTK v2.3 Release Notes

## Highlights of this Release
- Better ONNX support.
- Improved C# API.
- OpenCV is not required to install CNTK but to use Tensorboard Image feature.

## API
### C# API
- Improved C# API with performance gains in training and evaluation. 
-	During training and evaluation, data batch can be created from single managed buffer with offset. This eases the burden to prepare data in C# code. 
-	Internally, data marshalling is done more efficiently than Release 2.2. Use of chatty FloatVector has been avoided during training and evaluation.
### C++
- Exported “PreorderTraverse” C++ API: use to search the graph based on the provided criteria.

## Operators
### Group convolution
- We added support for group convolution on the GPU, exposed by C++ and Python API.
### Free static axes (FreeDimension) support for more operators
- We have added free static axes support for additional operators such as pooling (MaxPool, AveragePool), global pooling, unpooling, and reshape. With this increased support, it should be possible to run most common convolutional pipelines (CNNs) with free static axes. 
### Backcompact
- Support loading v1 model with DiagTimes OP.

## Performance
### Convolution with free static axes support
-We have improved the training performance for models that use convolution operation with free static axes support. For certain models, we see training speed up of more than x5. 

## ONNX
- Improved ONNX support in CNTK.
- Update ONNX to the latest ONNX from https://github.com/onnx/onnx
- Cover most vision model such as Resnet, Inception and VGG (Only model saved in V2 CNTK format).
- Fix a lot of bugs.

## Deprecated
### Support for Python 3.4 will be removed from CNTK releases later than v2.3.
