 This file contains step-by-step instructions on how to build the Python API for CNTK on Windows.

# no support yet for python 2.7
# recommended python version is 3.4 (with numpy & scipy)
# supported platform: 64 bit

# Set up compiler and its variant

call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall" amd64

set MSSdk=1
set DISTUTILS_USE_SDK=1

# Make sure swig.exe is in your path. For example, if SWIG is installed to c:\local\swigwin-3.0.10, run the following:

set PATH=c:\local\swigwin-3.0.10;%PATH%

# a) If you are just building to use it locally:
    # Build -> generate .pyd

    # 1) Run the following:
    python .\setup.py build_ext --inplace --force

    # 2) add to PATH the path to CNTK dlls (e.g. in ..\..\x64\Release)
        set PATH=%CD%\..\..\x64\Release;%PATH%
    # 3) Add to PYTHONPATH the local path and the path to the Python examples
        set PYTHONPATH=%CD%;%CD%\examples;%PYTHONPATH%
    # 4) test by running any of the examples or running py.test from inside the cntk\tests or cntk\ops\tests directories.

# b) If you want to package it:
    # 1) install the following:
    pip install twine wheel

    # 2) Run the following (note: --inplace not required when packaging):
    python .\setup.py build_ext bdist_wheel

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
        # Try to run any of the examples, some examples come up with a script that fetches and prepares the data,
        # other examples use data files that are checked in inside the cntk repository.
    
