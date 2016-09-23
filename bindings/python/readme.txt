# This file contains step-by-step instructions on how to build the Python API for CNTK

# no support yet for python 2.7
# recommended python version is 3.4 (with numpy & scipy)
# supported platform: 64 bit

# Set up compiler and its variant

SET PATH=%PATH%;C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC;C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin
SET MSSdk=1
SET DISTUTILS_USE_SDK=1
vcvarsall amd64

# Generate .cxx and .py out of .i. Please check the path to the SwigWin binaries inside swig.bat
# run swig.bat from within swig folder
swig.bat


# a) If you are just building to use it locally:
    # Build -> generate .pyd
    # 1) go two levels up
    # 2) run the following:
    python .\setup.py build_ext -if -c msvc --plat-name=win-amd64

    # 3) add to PATH the path to cntk dlls (e.g. e:\CNTK\x64\Release)
    # 4) add to PYTHONPATH the path to the python api source (e.g. e:\CNTK\bindings\python\)
    # 5) test by running any of the examples or running py.test from the inside bindings\python directory

# b) If you want to package it:
    # 1) install the following:
    pip install twine
    pip install wheel

    # 2) go two levels up & run:
    python .\setup.py build_ext -if -c msvc --plat-name=win-amd64 bdist_wheel

    # 3) put the wheel file on some http server

    # 4) from your machine, run pip install
    pip install http://your-url:your-port/cntk-0.0.0-cp34-cp34m-win_amd64.whl

    # 5) check that it is loaded correctly
    python
    >>> import cntk
    
    # 6) Running examples:
        # Clone the python examples folder from the CNTK repository and add its path to PYTHONPATH    
        # (e.g. setx PYTHONPATH %PYTHONPATH%;C:\work\cntk\bindings\python\examples in an Admin shell,
        # or rather setx PYTHONPATH C:\work\cntk\bindings\python\examples if no PYTHONPATH defined yet).
        # Try to run any of the examples, some examples come up with s script that fetches and prepares the data,
        # other examples use data files that are checked in inside the cntk repository.
    
