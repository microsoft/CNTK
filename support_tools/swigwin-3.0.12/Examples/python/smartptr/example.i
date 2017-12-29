/* File : example.i */
%module example

%{
#include "example.h"
#include "smartptr.h"
%}

/* Let's just grab the original header file here */
%include "example.h"

/* Grab smart pointer template */

%include "smartptr.h"

/* Instantiate smart-pointers */

%template(ShapePtr) SmartPtr<Shape>;


