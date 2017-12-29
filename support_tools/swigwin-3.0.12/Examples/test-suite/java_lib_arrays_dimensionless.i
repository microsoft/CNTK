%module java_lib_arrays_dimensionless

%include "arrays_java.i"

// Can't wrap dimensionless arrays, so we use the old pointer approach
%apply SWIGTYPE* { int globalints[], int constglobalints[], int Bar::ints[] }

// Test %apply for arrays in arrays_java.i library file
%apply bool []                  { bool *array }
%apply char []                  { char *array }
%apply signed char []           { signed char *array }
%apply unsigned char []         { unsigned char *array }
%apply short []                 { short *array }
%apply unsigned short []        { unsigned short *array }
%apply int []                   { int *array }
%apply unsigned int []          { unsigned int *array }
%apply long []                  { long *array }
%apply unsigned long []         { unsigned long *array }
%apply long []                  { long *array }
%apply unsigned long long []    { unsigned long long *array }
%apply float []                 { float *array }
%apply double []                { double *array }

%include "arrays_dimensionless.i"


