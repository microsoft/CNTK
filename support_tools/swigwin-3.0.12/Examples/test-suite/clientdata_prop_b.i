/* This file tests the clientdata propagation at swig wrapper
   generation.  It tests a bug in the TCL module where the
   clientdata was not propagated correctly to all classes */

%module clientdata_prop_b

%{
#include "clientdata_prop_b.h"
%}

%import "clientdata_prop_a.i"

%include "clientdata_prop_b.h"

%types(tA *);

%newobject new_t2A;
%newobject new_t3A;
%newobject new_tD;
%newobject new_t2D;
