/* File : example.i */
%module example

/* This example illustrates a couple of different techniques
   for manipulating C pointers */

/* First we'll use the pointer library */
extern void add(int *x, int *y, int *result);
%include pointer.i

/* Next we'll use some typemaps */

%include typemaps.i
extern void sub(int *INPUT, int *INPUT, int *OUTPUT);

/* Next we'll use typemaps and the %apply directive */

%apply int *OUTPUT { int *r };
extern int divide(int n, int d, int *r);




