/* This file tests the clientdata propagation at swig wrapper
   generation.  It tests a bug in the TCL module where the
   clientdata was not propagated correctly to all classes */

%module clientdata_prop_a
%{
  #include "clientdata_prop_a.h"
%}

%include "clientdata_prop_a.h"

%newobject new_tA;
