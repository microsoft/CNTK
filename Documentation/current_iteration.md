# CNTK v2.3 Release Notes

### Highlights of this Release
- OpenCV is not required to install CNTK but to use Tensorboard Image feature.

### Python-binding for CNTK
Support for Python 3.4 will be removed from CNTK releases later than v2.3.

## Operators
### Group convolution
 -We added support for group convolution on the GPU, exposed by C++ and Python API. It is useful for models such as the original Alexnet. 

## Performance
### Convolution with free static axes support
-We have improved the training performance for models that use convolution operation with free static axes support. For certain models, we see training speed up of more than x5. 
