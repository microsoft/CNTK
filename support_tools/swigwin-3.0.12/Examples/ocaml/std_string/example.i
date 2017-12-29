/* -*- C++ -*- */
/* File : example.i -- stolen from the guile std_vector example */
%module example

%{
#include "example.h"
%}

%include stl.i

/* Let's just grab the original header file here */
%include "example.h"
