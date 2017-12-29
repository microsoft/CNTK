%module port

%include guilemain.i

/* Include the required FILE * typemaps */
%include ports.i

%{
#include <stdio.h>
%}

%inline %{
void print_int(FILE *f, int i);
int read_int(FILE *f);
%}
