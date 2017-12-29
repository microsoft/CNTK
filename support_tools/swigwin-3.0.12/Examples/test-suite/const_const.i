/* This interface file tests whether SWIG handles types like
   "const int *const" right.

   SWIG 1.3a5 signals a syntax error.
*/

%module const_const

%typemap(in) const int *const { $1 = NULL; }

%inline %{
void foo(const int *const i) {}
%}
