/* File : example.i */
%module Example

%inline %{
extern int    gcd(int x, int y);
extern double Foo;
%}
