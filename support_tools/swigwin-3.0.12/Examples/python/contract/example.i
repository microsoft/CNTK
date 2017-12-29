/* File : example.i */
%module example

%contract gcd(int x, int y) {
require:
	x >= 0;
	y >= 0;
}

%contract fact(int n) {
require:
	n >= 0;
ensure:
	fact >= 1;
}

%inline %{
extern int    gcd(int x, int y);
extern int    fact(int n);
extern double Foo;
%}
