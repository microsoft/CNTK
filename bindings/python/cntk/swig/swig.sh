#!/bin/bash

SWIG=${SWIG-swig-3.0.10/root/bin/swig}

if [[ -x "$SWIG" ]]; then
  ${SWIG} -c++ -python -I../../../../Source/CNTKv2LibraryDll/API/ cntk_py.i
  mv cntk_py.py ..

else
  printf "Error: Cannot find executable at '%s'\n" $SWIG
  printf "       Please install swig (>= 3.0.10), and let the \$SWIG environment\n"
  printf "       variable to point to the SWIG binary location.\n"
fi
