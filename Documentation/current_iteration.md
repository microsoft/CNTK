# CNTK Current Iteration

## Highlights of this release
* Moved to CUDA 10 for both Windows and Linux.
* Support advance RNN loop in ONNX export.
* Export larger than 2GB models in ONNX format.

## CNTK support for CUDA 10

CNTK now supports CUDA 10. This requires an update to build environment to Visual Studio 2017 v15.9 for Windows.

To setup build and runtime environment on Windows:
* Install [Visual Studio 2017](https://www.visualstudio.com/downloads/). Note: going forward for CUDA 10 and beyond, it is no longer required to install and run with the specific VC Tools version 14.11.
* Install [Nvidia CUDA 10](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64)
* From PowerShell, run:
    [DevInstall.ps1](../Tools/devInstall/Windows/DevInstall.ps1)
* Start Visual Studio 2017 and open [CNTK.sln](./CNTK.sln).

To setup build and runtime environment on Linux using docker, please build Unbuntu 16.04 docker image using Dockerfiles [here](./Tools/docker). For other Linux systems, please refer to the Dockerfiles to setup dependent libraries for CNTK.