# Multiverso Python/Theano/Lasagne Binding


## Introduction
Multiverso is a parameter server framework for distributed machine learning. This package can leverage multiple machines and GPUs to speed up the python programs.


## Installation

1. (For GPU support only) Install CUDA, cuDNN according to this [guide](https://github.com/Microsoft/fb.resnet.torch/blob/multiverso/INSTALL.md). You just need finish the steps before [Install Torch](https://github.com/Microsoft/fb.resnet.torch/blob/multiverso/INSTALL.md#install-torch).
1. Install the multiverso
    * On linux: Please follow the [README](https://github.com/Microsoft/multiverso/blob/master/README.md#build) to build and install multiverso.
    * On windows: You need MSBuild.exe installed and make sure your system can find it in the $PATH. Then you should run [build_dll.bat](https://github.com/Microsoft/multiverso/blob/master/src/build_dll.bat) to build the .dll file and install the .dll. There isn't auto-installer for windows now, so you have to copy the .dll to either system $PATH or the multiverso package folder.
1. Install the requirements
    * `gfortran` is required by scipy. e.g. you can install it by `sudo apt-get install gfortran` on ubuntu.
    * (Optional) You need python-nose to run the unit tests. e.g. you can install it by `sudo apt-get install python-nose` on ubuntu.
1. Install python binding with the command `sudo python setup.py install`


## Run Unit Tests
```
nosetests
```


## Documentation
* [Tutorial](https://github.com/Microsoft/multiverso/wiki/How-to-write-python-code-with-multiverso)
* Api documents are written as docstrings in the python source code.
* [Benchmark](https://github.com/Microsoft/multiverso/wiki/Multiverso-Python-Binding-Benchmark)
