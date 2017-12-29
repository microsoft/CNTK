/* File : example.i */
%module example
#pragma SWIG nowarn=SWIGWARN_IGNORE_OPERATOR_EQ
%{
#include "example.h"
%}

/* This header file is a little tough to handle because it has overloaded
   operators and constructors.  We're going to try and deal with that here */

/* This turns the copy constructor in a function ComplexCopy() that can
   be called */

%rename(assign) Complex::operator=;
%rename(plus) Complex::operator+;
%rename(minus) Complex::operator-(const Complex &) const;
%rename(uminus) Complex::operator-() const;
%rename(times) Complex::operator*;

/* Now grab the original header file */
%include "example.h"

/* An output method that turns a complex into a short string */
%extend Complex {
 char *toString() {
   static char temp[512];
   sprintf(temp,"(%g,%g)", $self->re(), $self->im());
   return temp;
 }
 static Complex* copy(const Complex& c) {
   return new Complex(c);
 }
};

