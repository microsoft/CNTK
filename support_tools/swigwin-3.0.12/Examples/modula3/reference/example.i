/* File : example.i */

/* This file has a few "typical" uses of C++ references. */

%module Example

%{
#include "example.h"
%}

%pragma(modula3) unsafe="1";

%insert(m3wrapintf) %{FROM ExampleRaw IMPORT Vector, VectorArray;%}
%insert(m3wrapimpl) %{FROM ExampleRaw IMPORT Vector, VectorArray;%}

%typemap(m3wrapretvar)  Vector %{vec: UNTRACED REF Vector;%}
%typemap(m3wrapretraw)  Vector %{vec%}
%typemap(m3wrapretconv) Vector %{vec^%}


/* This helper function calls an overloaded operator */
%inline %{
Vector addv(const Vector &a, const Vector &b) {
  return a+b;
}
%}

%rename(Vector_Clear) Vector::Vector();
%rename(Add) Vector::operator+;
%rename(GetItem) VectorArray::operator[];

%include "example.h"
