
# CNTK v2.3 Release Notes

## Highlights of this Release
- Better ONNX support.
- Switching to NCCL2 for better perfomacne in distributed training.
- Improved C# API.
- OpenCV is not required to install CNTK, it is only required forTensorboard Image feature and image reader.
- Various performance improvement.

## API
### C# API
- Improved C# API with performance gains in training and evaluation. 
-	During training and evaluation, data batch can be created from single managed buffer with offset. This eases the burden to prepare data in C# code. 
-	Internally, data marshalling is done more efficiently than Release 2.2. Use of chatty FloatVector has been avoided during training and evaluation.
### C++
- Exported “PreorderTraverse” C++ API: use to search the graph based on the provided criteria.
### Python and C++
- Add custom attributes to primitive function, which would be serialized/deserialized when model save/load. 
- Some usage:
```python
    func = C.plus(a, b)
    func.custom_attributes = {'test':'abc', 'dict':{'a':1, 'b':2}, 'list':[1,2,3]} 
    func.custom_attributes['test2'] = 'def'
```

## Operators
### Group convolution
- We added support for group convolution on the GPU, exposed by C++ and Python API.
### Free static axes (FreeDimension) support for more operators
- We have added free static axes support for additional operators such as pooling (MaxPool, AveragePool), global pooling, unpooling, and reshape. With this increased support, it should be possible to run most common convolutional pipelines (CNNs) with free static axes. 
### Backcompact
- Support loading v1 model with DiagTimes OP.

## Performance
### Convolution with free static axes support
- We have improved the training performance for models that use convolution operation with free static axes support. For certain models, we see training speed up of more than x5. 
### Validation Performance
- Improve validation performance and remove a lot of unneeded validation check.
### CPU Convolution
- Move CPU Convolution to use MKL-ML, which leads to ~4x speedup in AlexNet training.
### Moving to NCCL2
- NCCL2 would be enabled by default in official CNTK releases for Linux GPU build, which reduced aggregation cost in distributed training. For Python users, there’s no impact as NCCL binary is included in the Linux Python wheels. For BrainScript users on Linux, they need to install [NCCL library]( https://github.com/NVIDIA/nccl) as part of CNTK environment setup, similar to CUDA and CUDNN. CPU builds and Windows builds are not affected since NCCL is available for Linux only.

## ONNX
- Improved ONNX support in CNTK.
- Update ONNX to the latest ONNX from https://github.com/onnx/onnx
- Cover most vision model such as Resnet, Inception and VGG (Only model saved in V2 CNTK format).
- Fix a lot of bugs.

## Dependencies
### Removed OpenCV dependency from CNTK core.
- CNTK 2.2 requires you to install OpenCV to use CNTK but it is optional for CNTK 2.3
- You need to install OpenCV only if you are planning to use ImageReader or TensorBoard’s Image feature.
### Upgraded ImageIO to 2.2.0
- Anaconda doesn’t support Python 3.4. CNTK will also remove Python 3.4 support in future releases.
### MKL
- Switch from CNTKCustomMKL to Intel MKLML. MKLML is released with [Intel MKL-DNN](https://github.com/01org/mkl-dnn/releases) as a trimmed version of Intel MKL for MKL-DNN. To set it up:

#### On Linux:
    sudo mkdir /usr/local/mklml
    sudo wget https://github.com/01org/mkl-dnn/releases/download/v0.11/mklml_lnx_2018.0.1.20171007.tgz
    sudo tar -xzf mklml_lnx_2018.0.1.20171007.tgz -C /usr/local/mklml

#### On Windows:
    Create a directory on your machine to hold MKLML, e.g. mkdir c:\local\mklml
    Download the file [mklml_win_2018.0.1.20171007.zip](https://github.com/01org/mkl-dnn/releases/download/v0.11/mklml_win_2018.0.1.20171007.zip).
    Unzip it into your MKLML path, creating a versioned sub directory within.
    Set the environment variable `MKLML_PATH` to the versioned sub directory, e.g. setx MKLML_PATH c:\local\mklml\mklml_win_2018.0.1.20171007

## Deprecated
### Support for Python 3.4 will be removed from CNTK releases later than v2.3.
