/* File : example.i */
%module example

%{
extern void add(double *, double *, double *);
extern void sub(int *, int *, int *);
extern int divide(int, int, int *);
%}

/* This example illustrates a couple of different techniques
   for manipulating C pointers */

%include phppointers.i
/* First we'll use the pointer library */
extern void add(double *REF, double *REF, double *REF);

/* Next we'll use some typemaps */

%include typemaps.i
extern void sub(int *INPUT, int *INPUT, int *OUTPUT);

/* Next we'll use typemaps and the %apply directive */

//%apply int *OUTPUT { int *r };
//extern int divide(int n, int d, int *r);





