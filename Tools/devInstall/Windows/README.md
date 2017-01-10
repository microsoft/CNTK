## Developer Setup Windows from a script

This documents describes the setup of the CNTK developer environment. The script follows overall the recipe described in the public documentation of the CNTK developer setup, available [here](https://github.com/Microsoft/CNTK/wiki/Setup-CNTK-on-Windows). 

>Note: Before startingt the script base installation, please check the CNTK Wiki for more detailed instructions. Make sure the prerequisites for the installation are satisfied on your system.

### The installation script

The installation script itself is written for the Powershell environment. By default Powershell doesn't allow the execution of any scripts. To allow execution of the installation script, start Powershell from a **standard Windows command shell** by:
```
start powershell -executionpolicy remotesigned
```
This will open a Powershell environment. In this enviroment change into the directory you copied the install script into, here you can directly start the installation script
```
cd c:\local\devinstall
.\devinstall.ps1
```
>Note: If you now receive an errormessage stating that running scripts is disabled on your system, please make sure you started powershell with the changed executionpolicy. This is enabling 
script execution in your created Powershell environment.

The script will inspect your system and determine the components which are missing to build the Microsoft Cognitive Toolkit. You will be notified about the proposed installation steps. At this point you are running in a **demo** mode - NO changes to your system are being performed. 

### Running the installation script

If you are satisfied with the proposed changes, you can proceed to the actual installation. The installation script supports several command line options. ``get-help .\install.ps1`` will give you a list over the available option. At this point it is recommended to start the installation by adding the `-Execute` parameter:
```
.\devinstall.ps1 -execute
```
The script will download needed components from the web, therefore a connection to the Internet is required. Downloaded components are stored in the directory ``c:\installCacheCntk`` and can be removed after complete installation. During execution of the installation script you might receive requests/warnings from UAC (User Account Control) depending on your system configuration. Please acknowledge the execution and installation of the downloaded components.

> Note: Some components of the installation script (i.e. NVidia CUDA install ) might performan a reboot of your system. You can just start the installation script again. It will analyze your system again and reuse already downloaded components and will only perform the actions necessary.

### Result of the installation script

The following changes (if necessary) to your system will be performed by the installation script:

- Installation of Microsoft MPI
- Installation of the Microsoft MPI SDK
- NVidia CuDNN 5.1 for CUDA 8
    - Location: ``c:\local\cudnn-8.0-v5.1``
    - Environment: ``CUDNN_PATH``
- NVidia CUB 1.4.1
    - Location: ``c:\local\cub-1.4.1``
    - Environment: ``CUB_PATH``
- Boost 1.60
    - Location: ``boost_1_60_0-msvc-14.0``
    - Environment: ``BOOST_INCLUDE_PATH``, ``BOOST_LIB_PATH``
- A CNTK specific MKL library
    - Location: ``c:\local\cntkmkl``
    - Environment: ``CNTK_MKL_PATH``
- OpenCV 3.1.0
    - Location: ``c:\local\Opencv3.1.0``
    - Environment: ``OPENCV_PATH_V31``
- Zlib Source
    - Location: ``c:\local\src\zlib-1.2.8``
- Libzip Source
    - Location: ``c:\local\src\libzip-1.1.3``
- Zlib - Precompiled version of Zlib and libzip for CNTK
    - Location: ``c:\local\zlib-vs15``
    - Environment: ``ZLIB_PATH``
- Protobuf 3.1.0 Source
    - Location: ``c:\local\src\protobuf-3.1.0``
- Protobuf 3.1.0 - Precompiled version for CNTK
    - Location: ``c:\local\protobuf-3.1.0-vs15``
    - Environment: ``PROTOBUF_PATH``
- SWIG 3.0.10
    - Location: ``c:\local\swigwin-3.0.10``
    - Environment: ``SWIG_PATH``
- Anaconda3 - 4.1.1
    - Location: ``c:\local\Anaconda3-4.1.1-Windows-x86_64``
    This is a local Anaconda install, it hasn't been added to the path, and is registered only to the current user

>Note: This script already installed the compiled Protobuf library, as well as the compiled zlib and libzip components. It isn't necessary to perform the additional compilation steps (listed in the public documentation) for these components.

Once the script finished, you should be ready to build. A couple of points to note:
 - Depending on the tools installed, the script might have done changes to your system. Rebooting you system might be a good choice.
 - It is possible to run the script multiple times. Powershell is very specific that it doesn't pick up changes to environment variables which are done in the script, you have to restart powershell to pick up the latest environment.
 - If you are planning on using a GPU, you should install the latest GPU driver from NVidia, and reboot your system.
 




