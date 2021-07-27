# Build on Windows Notes
* Clone recursive
```
git clone --recursive https://github.com/nietras/CNTK
```
>* Download CUDA Toolkit 10.0 and install (don't install driver since old!)
>  https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork
* Download CUDA Toolkit 10.2 plus patches and install (don't install driver since old!)
  https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork
* To support VS2019 fix `C:\git\oss\CNTK\Tools\devInstall\Windows\helper\Operations.ps1` 
  * `OpCheckVS2017` => `OpCheckVS2019`
* Run `C:\git\oss\CNTK\Tools\devInstall\Windows\DevInstall.ps1`
```
                                                                                                                                                                                                                                                                                                                                                                        This script will setup the CNTK Development Environment on your machine.                                                                                                            More help is given by calling get-help .\devInstall.ps1                                                                                                                                                                                                                                                                                                                 The script will analyse your machine and will determine which components are required.                                                                                              The required components will be downloaded into [c:\installCacheCntk]                                                                                                               Repeated operation of this script will reuse already downloaded components.                                                                                                                                                                                                                                                                                             
1 - I agree and want to continue
Q - Quit the installation process

1
Determining Operations to perform. This will take a moment...
Scan System for installed programs
Checking for Visual Studio 2019
Checking for NVidia Cuda 10.0
Checking for NVidia CUDNN 7.3.1 for CUDA 10.0 in c:\local\cudnn-10.0-v7.3.1
Checking for NVidia CUB 1.8.0 in c:\local\cub-1.8.0
Checking for CMake 3.6.2 in C:\Program Files\cmake\bin
Checking for installed MSMPI 70
Checking for installed MSMPI 70 SDK
Checking for Boost 1.60.0 in c:\local\boost_1_60_0-msvc-14.0
Checking for MKLML and MKL-DNN 0.12 CNTK Prebuild in c:\local\mklml-mkldnn-0.12
Checking for SWIG 3.0.10 in c:\local\swigwin-3.0.10
Checking for ProtoBuf 3.1.0 Source in c:\local\src\protobuf-3.1.0
Checking for ProtoBuf 3.1.0 VS17 CNTK Prebuild in c:\local\protobuf-3.1.0-vs17
Checking for zlib / libzip from source in c:\local\src
Checking for ZLib VS17 CNTK Prebuild in c:\local\zlib-vs17
Checking for OpenCV-3.1 in c:\local\Opencv3.1.0
Checking for Anaconda3-4.1.1 in c:\local\Anaconda3-4.1.1-Windows-x86_64
Checking for Python 35 Environment in c:\local\Anaconda3-4.1.1-Windows-x86_64
Checking pre-requisites finished


The following operations will be performed:
 * Installing NVidia CUDNN 7.3.1 for CUDA 10.0
 * Installing NVidia CUB 1.8.0
 * Installing CMake 3.6.2
 * Installing MSMPI 70
 * Installing MSMPI 70 SDK
 * Installing Boost 1.60.0
 * Installing MKLML and MKL-DNN 0.12 CNTK Prebuild
 * Installing SWIG 3.0.10
 * Installing ProtoBuf 3.1.0 Source
 * Installing ProtoBuf 3.1.0 VS17 CNTK Prebuild
 * Installing zlib / libzip from source
 * Installing ZLib VS17 CNTK Prebuild
 * Installing OpenCV-3.1
 * Installing Anaconda3-4.1.1
 * Creating Python 35 Environment

Do you want to continue? (y/n)
y
Performing download operations
Downloading [https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/cudnn/7.3.1/cudnn-10.0-windows10-x64-v7.3.1.20.zip], please be patient....
Downloading [https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/cub/1.8.0/cub-1.8.0.zip], please be patient....
Downloading [https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/cmake/3.6.2/cmake-3.6.2-win64-x64.msi], please be patient....
Downloading [https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/msmpi/70/MSMpiSetup.exe], please be patient....
Downloading [https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/msmpisdk/70/msmpisdk.msi], please be patient....
Downloading [https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/boost/1.60.0/boost_1_60_0-msvc-14.0-64.exe], please be patient....
Downloading [https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/mkl-dnn/0.12/mklml-mkldnn-0.12.zip], please be patient....
Downloading [https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/swig/3.0.10/swigwin-3.0.10.zip], please be patient....
Downloading [https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/protobuf/3.1.0/protobuf-3.1.0.zip], please be patient....
Downloading [https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/protobuf/3.1.0/protobuf-3.1.0-vs17.zip], please be patient....
Downloading [https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/zlib/1.2.8/zlib128.zip], please be patient....
Downloading [https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/libzip/1.1.3/libzip-1.1.3.tar.gz], please be patient....
Downloading [https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/zlib/vs17/zlib-vs17.zip], please be patient....
Downloading [https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/opencv/3.1.0/opencv-3.1.0.exe], please be patient....
Downloading [https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/anaconda3/4.1.1/Anaconda3-4.1.1-Windows-x86_64.exe], please be patient....
Download operations finished

Performing install operations
Installing NVidia CUDNN 7.3.1 for CUDA 10.0
Installing NVidia CUB 1.8.0
Installing CMake 3.6.2
Installing MSMPI 70
Installing MSMPI 70 SDK
Installing Boost 1.60.0
Installing MKLML and MKL-DNN 0.12 CNTK Prebuild
Installing SWIG 3.0.10
Installing ProtoBuf 3.1.0 Source
Installing ProtoBuf 3.1.0 VS17 CNTK Prebuild
Installing zlib / libzip from source
Installing ZLib VS17 CNTK Prebuild
Installing OpenCV-3.1
Installing Anaconda3-4.1.1
.... This will take some time. Please be patient ....
Creating Python 35 Environment
Exception caught - function main / failure
System.Management.Automation.RuntimeException: Running [start-process  env create --file "C:\git\oss\CNTK\scripts\install\windows\conda-windows-cntk-py35-environment.yml" --prefix c:\local\Anaconda3-4.1.1-Windows-x86_64\envs\cntk-py35] failed with exit code [1]
```
* Visual Studio Installer
  * **Windows 10 SDK (10.0.16299.0)**
  * **MSVC v141  VS 2017 C++ x64/86 build tools (v14.16)**
* Set Environment Variables
```
setx CNTK_ENABLE_ASGD false
setx MKL_PATH c:\local\mklml-mkldnn-0.12
setx BOOST_INCLUDE_PATH c:\local\boost_1_60_0-msvc-14.0
setx BOOST_LIB_PATH c:\local\boost_1_60_0-msvc-14.0\lib64-msvc-14.0
setx PROTOBUF_PATH c:\local\protobuf-3.1.0-vs17
setx CUDNN_PATH c:\local\cudnn-10.0-v7.3.1\cuda
setx OPENCV_PATH_V31 c:\local\Opencv3.1.0\build
setx ZLIB_PATH c:\local\zlib-vs17
setx CUB_PATH c:\local\cub-1.8.0\
setx SWIG_PATH C:\local\swigwin-3.0.10
```
* Restart the shell afterwards or use `set` not `setx`
* Follow https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-with-script-on-windows
* Then https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-windows#building-cntk

# CUDA 10.2 - First step to see if can build
Focus on building `Common` project first.


`CUDA_PATH_V10_2` and related are pre-defined environment variables.

Fix warning treated as error in `GPUMatrix.cu`, `GPUSparseMatrix.cu`, `GPUTensor.cu` by inserting:
```
#pragma warning(disable : 4324) // 'thrust::detail::aligned_type<2>::type': structure was padded due to alignment
```
at the top of the file.

See https://github.com/nietras/CNTK/pull/1/files

# CUDA 11 - Modifications to support Ampere GPUs
https://github.com/microsoft/CNTK/issues/3835

## Look at changes in other repo for reference
https://github.com/haryngod/CNTK/tree/2.7-cuda-11.1

