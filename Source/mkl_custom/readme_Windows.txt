<review - and extend>

In this directory you find the files necessary to create a custom MKL library for CNTK.

By default CNTK uses a prebuild custom MKL library. If you install the CNTK binaries, this custom DLL is included in the binary download.
In case you want to build CNTK by yourself, you will have to download and install a custom MKL libaray onto your local machine. 
A detailed describtion is part of the CNTK wiki, see https://github.com/Microsoft/CNTK/wiki/Setup-CNTK-on-your-machine

If you want to add new MKL functions used by CNTK you will have to build your own custom MKL library. This requires you to install the Intel MKL SDK
(https://software.intel.com/en-us/intel-mkl/) and follow the steps in the MKL documentation for custom link libraries (https://software.intel.com/en-us/node/528366)

The file 'cntklist.txt' in this directory describes the MKL functions used by CNTK. With an installed Intel MKL SDK you copy this file into the 
builder directory, add the additional MKL functionality you want to use from CNTK, and following the steps in the MKL documentation you can build your own 
custom MKL library by calling: 
    nmake libintel64 export=cntklist.txt

This will create a new MKL_CUTOM.DLL you can use in your own development