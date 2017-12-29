/* File : example.i */
%module example

%inline %{
extern int    gcd(int x, int y);
extern double Foo;
%}
