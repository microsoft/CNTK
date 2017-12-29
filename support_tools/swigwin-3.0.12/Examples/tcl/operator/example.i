/* File : example.i */
%module example

%{
#include "example.h"
%}

/* This header file is a little tough to handle because it has overloaded
   operators and constructors.  We're going to try and deal with that here */

/* This turns the copy constructor in a function ComplexCopy() that can
   be called */

%rename(ComplexCopy) Complex::Complex(Complex const &);

/* Now grab the original header file */
%include "example.h"

/* An output method that turns a complex into a short string */
%extend Complex {
   char *str() {
       static char temp[512];
       sprintf(temp,"(%g,%g)", $self->re(), $self->im());
       return temp;
   }
};


