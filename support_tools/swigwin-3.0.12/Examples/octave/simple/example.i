/* File : example.i */
%module swigexample

%feature("autodoc", 1);

%inline %{
extern int    gcd(int x, int y);
extern double Foo;
%}
