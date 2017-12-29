/* File : example.i */
%module swigexample

%feature("autodoc", 1);

%{
#include "example.h"
%}

/* Let's just grab the original header file here */

%include "example.h"
