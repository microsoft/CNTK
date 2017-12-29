/* File : example.i */
%module example
%{
#include "example.h"
%}

/* Wrap a function taking a pointer to a function */
extern int  do_op(int a, int b, int (*op)(int, int));

/* Now install a bunch of "ops" as constants */
%callback("%(upper)s");
int add(int, int);
int sub(int, int);
int mul(int, int);
%nocallback;

extern int (*funcvar)(int,int);

