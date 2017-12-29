/* File : example.i */

/* Basic C example for swig contract */
/* Tiger, University of Chicago, 2003 */

%module example

%contract Circle (int x, int y, int radius) {
require:
     x      >= 0;
     y      >= 0;
     radius >  x;
ensure:
     Circle >= 0;
}

%inline %{
extern int Circle (int x, int y, int radius);
%}
