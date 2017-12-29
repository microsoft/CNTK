/* File : example.i */
%module example

%{
#include "example.h"
%}

/* Force the generated Java code to use the C enum values rather than making a JNI call */
%javaconst(1);

/* Let's just grab the original header file here */

%include "example.h"

