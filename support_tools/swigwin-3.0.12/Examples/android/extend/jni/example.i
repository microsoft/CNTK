/* File : example.i */
%module(directors="1") example
%{
#include "example.h"
%}

%include "std_vector.i"
%include "std_string.i"

/* turn on director wrapping for Manager */
%feature("director") Employee;
%feature("director") Manager;

/* A base class for callbacks from C++ to output text on the Java side */
%feature("director") Streamer;

%include "example.h"

