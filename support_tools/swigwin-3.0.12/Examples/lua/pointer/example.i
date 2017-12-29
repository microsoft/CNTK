/* File : example.i */
%module example

%{
   #include <stdlib.h>
%}

/* This example illustrates a couple of different techniques
   for manipulating C pointers */

/* First we'll use the pointer library */
%inline %{
extern void add(int *x, int *y, int *result);
%}
%include cpointer.i
%pointer_functions(int, intp);

/* Next we'll use some typemaps */

%include typemaps.i
extern void sub(int *INPUT, int *INPUT, int *OUTPUT);
%{
extern void sub(int *, int *, int *);
%}

/* Next we'll use typemaps and the %apply directive */

%apply int *OUTPUT { int *r };
%inline %{
extern int divide(int n, int d, int *r);
%}




