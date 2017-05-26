# CNTK custom MKL

This directory contains the necessary files to create a custom Intel® Math Kernel Library (MKL)
for usage by CNTK ("CNTK custom MKL" for short).

By default, a CNTK binary with Intel® MKL support includes a prebuilt CNTK
custom MKL.
If you want to build CNTK with Intel® MKL support yourself, you can install a
prebuilt CNTK custom MKL, available for download from the [CNTK web site](https://www.cntk.ai/mkl).
See [CNTK's setup instructions](https://github.com/Microsoft/CNTK/wiki/Setup-CNTK-on-your-machine)
for more details.

If you want to add new Intel® MKL functions to be used by CNTK you will have to
build your own CNTK custom MKL.
This requires you to install the [Intel MKL SDK](https://software.intel.com/en-us/intel-mkl/) for your platform.
Then, in this directory,
* extend the file `headers.txt` to expose new headers,
* extend the file `functions.txt` to expose new functions, and
* use `build-linux.sh` or `build-windows.cmd` to build for your platform.

For further documentation please see the Developer Guide for the Intel® MKL, in particular
[Building Custom Shared Objects (Linux)](https://software.intel.com/en-us/node/528533) and
[Building Custom Dynamic-link Libraries (Windows)](https://software.intel.com/en-us/node/528362).
