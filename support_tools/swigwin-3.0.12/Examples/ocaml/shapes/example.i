/* File : example.i */
%module(directors="1") example
#ifndef SWIGSEXP
%{
	#include "example.h"
%}
#endif

%feature("director");
%include "example.h"
