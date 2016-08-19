# Set up compiler and its variant

SET PATH=%PATH%;C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC;C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin
SET MSSdk=1
SET DISTUTILS_USE_SDK=1
vcvarsall amd64

# Generate .cxx and .py out of .i. Please check the path to the SwigWin binaries inside swig.bat
swig.bat

# Copy cntk_py.py one level up.

# Build -> generate .pyd
# go one level up
python .\setup.py build_ext -if -c msvc --plat-name=win-amd64

# Run
python cntk_py_tester.py


