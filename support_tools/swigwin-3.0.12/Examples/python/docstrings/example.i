/* File : example.i */
%module example

%{
#include "example.h"
%}

/* %feature("docstring") has to come before the declaration of the method to
 * SWIG. */
%feature("docstring") Foo::bar "No comment"

/* Let's just grab the original header file here */
%include "example.h"

