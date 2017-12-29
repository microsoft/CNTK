%module example
%{
#include <math.h>
%}

/* File : example.i */
%module example

%contract cos(double d) {
require:
	d >= -3.14159265358979323845254338327950;
	d < 3.14159265358979323846264338327950;
ensure:
	cos >= -1.0;
	cos <= 1.0;
}

double cos(double d);