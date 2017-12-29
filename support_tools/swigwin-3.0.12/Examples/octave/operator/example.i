/* File : example.i */
%module swigexample

%feature("autodoc", 1);

#pragma SWIG nowarn=SWIGWARN_IGNORE_OPERATOR_EQ

%{
#include "example.h"
%}

/* Rename friend operator */
%rename(op_scalar_mul_ComplexVal) operator*(const double&, const ComplexVal&);

/* Now grab the original header file */
%include "example.h"

/* An output method that turns a complex into a short string */
%extend ComplexVal {
   char *__str__() {
       static char temp[512];
       sprintf(temp,"(%g,%g)", $self->re(), $self->im());
       return temp;
   }

   ComplexVal __paren__(int j) {
     return ComplexVal($self->re()*j,$self->im()*j);
   }

   double __float__() {
     return (double) $self->re();
   }

   double __real__() {
     return $self->re();
   }
   double __imag__() {
     return $self->im();
   }
   double __abs__() {
     return sqrt($self->re()*$self->re() + $self->im()*$self->im());
   }
   ComplexVal __conj__() {
     return ComplexVal($self->re(), -$self->im());
   }
   ComplexVal __exp__() {
     return exp($self->re()) * ComplexVal(cos($self->im()), sin($self->im()));
   }

};
