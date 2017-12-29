/* File : example.i */
%module example

%{
#include "example.h"
%}


/* Let's just grab the original header file here */

%include "example.h"

