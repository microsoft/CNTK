%module constant_expr;
/* Tests of constant expressions. */

%inline %{

/* % didn't work in SWIG 1.3.40 and earlier. */
const int X = 123%7;
#define FOO 12 % 9
double d_array[12 % 9];

%}
