/* File : example.i */
%module example

%{
#include "example.h"
%}

%include "std_string.i"

/* Let's just grab the original header file here */
%include "example.h"

