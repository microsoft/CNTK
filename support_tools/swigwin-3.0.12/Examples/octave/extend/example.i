/* File : example.i */
%module(directors="1") swigexample

%feature("autodoc", 1);

%{
#include "example.h"
%}

%include "std_vector.i"
%include "std_string.i"

/* turn on director wrapping for Manager */
%feature("director") Employee;
%feature("director") Manager;

%include "example.h"
