/* File : example.i */
%module example
%{
#include "example.h"
%}

/* Wrap a function taking a pointer to a function */
extern int  do_op(int a, int b, int (*op)(int, int));

extern int (*funcvar)(int,int);

