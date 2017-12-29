/* File : example.i */
%module example

%{
#include "example.h"
%}

%include "std_string.i"

// we want to return Exc objects to the interpreter
// therefore we add this typemap
// note: only works if Exc is copyable
%apply SWIGTYPE EXCEPTION_BY_VAL {Exc};

/* Let's just grab the original header file here */
%include "example.h"

