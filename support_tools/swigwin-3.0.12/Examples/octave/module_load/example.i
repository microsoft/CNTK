/* File : example.i */
/* module name given on cmdline */

%feature("autodoc", 1);

%{
#include "example.h"
%}

extern "C" int ivar;

int ifunc();
