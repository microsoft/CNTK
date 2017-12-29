/* File : example.i */
%module example

%warnfilter(SWIGWARN_IGNORE_OPERATOR_EQ);


%{
#include "example.h"
%}

/* This header file is a little tough to handle because it has overloaded
   operators and constructors.  We're going to try and deal with that here */

/* Grab the original header file */
%include "example.h"

/* An output method that turns a complex into a short string */
%extend Complex {
   char *__str__() {
       static char temp[512];
       sprintf(temp,"(%g,%g)", $self->re(), $self->im());
       return temp;
   }
};


