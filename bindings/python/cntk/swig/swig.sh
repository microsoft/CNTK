#!/bin/bash

# Please change this
SWIG=swig-3.0.10/root/bin/swig

${SWIG} -c++ -python -I../../../../Source/CNTKv2LibraryDll/API/ cntk_py.i

