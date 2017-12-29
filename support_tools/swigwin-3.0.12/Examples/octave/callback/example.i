/* File : example.i */
%module(directors="1") swigexample

%feature("autodoc", 1);

%{
#include "example.h"
%}

/* turn on director wrapping Callback */
%feature("director") Callback;

%include "example.h"
