/* File : example.i */
%module example

%{
#include "example.h"
%}

%include stl.i
/* instantiate the required template specializations */
namespace std {
    %template(IntVector)    vector<int>;
    %template(DoubleVector) vector<double>;
}

/* Let's just grab the original header file here */
%include "example.h"

