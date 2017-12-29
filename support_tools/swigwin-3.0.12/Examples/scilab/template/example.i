/* File : example.i */
%module example

%{
#include "example.h"
%}

/* Let's just grab the original header file here */
%include "example.h"

/* Now instantiate some specific template declarations */

%template(maxint) max<int>;
%template(maxdouble) max<double>;
%template(vecint) vector<int>;
%template(vecdouble) vector<double>;

