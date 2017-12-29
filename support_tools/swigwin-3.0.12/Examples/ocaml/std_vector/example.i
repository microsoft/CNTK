/* -*- C++ -*- */
/* File : example.i -- stolen from the guile std_vector example */
%module example

%{
#include "example.h"
%}

%include stl.i
/* instantiate the required template specializations */
%template(IntVector)    std::vector<int>;
%template(DoubleVector) std::vector<double>;

/* Let's just grab the original header file here */
%include "example.h"
