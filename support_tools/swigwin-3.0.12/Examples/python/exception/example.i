/* File : example.i */
%module example

%{
#include "example.h"
%}

%include "std_string.i"

/* Let's just grab the original header file here */
%include "example.h"

%inline %{
// The -builtin SWIG option results in SWIGPYTHON_BUILTIN being defined
#ifdef SWIGPYTHON_BUILTIN
bool is_python_builtin() { return true; }
#else
bool is_python_builtin() { return false; }
#endif
%}

