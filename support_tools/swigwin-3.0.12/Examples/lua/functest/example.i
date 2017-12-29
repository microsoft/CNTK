/* File : example.i */
%module example

%include "typemaps.i"

%inline %{
extern int add1(int x, int y);              /* return x+y -- basic function test */
extern void add2(int x, int *INPUT, int *OUTPUT); /* *z = x+*y  -- argin and argout test */
extern int add3(int x, int y, int *OUTPUT);    /* return x+y, *z=x-y -- returning 2 values */
extern void add4(int x, int *INOUT);        /* *y += x    -- INOUT dual purpose variable */
%}

