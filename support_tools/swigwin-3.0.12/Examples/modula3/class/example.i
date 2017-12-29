/* File : example.i */
%module Example

%{
#include "example.h"
%}

%insert(m3makefile) %{template("../swig")
cxx_source("example_wrap")%}

%typemap(m3rawinmode)    Shape *, Circle *, Square * ""
%typemap(m3rawrettype)   Shape *, Circle *, Square * "$1_basetype"

%typemap(m3wrapinmode)   Shape *, Circle *, Square * ""
%typemap(m3wrapargraw)   Shape *, Circle *, Square * "self.cxxObj"

%typemap(m3wrapretvar)   Circle *, Square * "cxxObj : ExampleRaw.$1_basetype;"
%typemap(m3wrapretraw)   Circle *, Square * "cxxObj"
%typemap(m3wrapretconv)  Circle *, Square * "NEW($1_basetype,cxxObj:=cxxObj)"
%typemap(m3wraprettype)  Circle *, Square * "$1_basetype"

/* Should work with and without renaming
%rename(M3Shape) Shape;
%rename(M3Circle) Circle;
%rename(M3Square) Square;
%typemap(m3wrapintype)   Shape *, Circle *, Square * "M3$1_basetype"
%typemap(m3wraprettype)  Shape *, Circle *, Square * "M3$1_basetype"
%typemap(m3wrapretconv)           Circle *, Square * "NEW(M3$1_basetype,cxxObj:=cxxObj)"
*/

/* Let's just grab the original header file here */
%include "example.h"
