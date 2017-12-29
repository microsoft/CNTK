/* File : example.i */
%module(directors="1") example
%{
#include "example.h"
%}

/* turn on director wrapping Callback */
%feature("director") Callback;

/* Caller::setCallback(Callback *cb) gives ownership of the cb to the
 * Caller object.  The wrapper code should understand this. */
%apply SWIGTYPE *DISOWN { Callback *cb }; 

%include "example.h"

