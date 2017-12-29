/* This test checks whether SWIG correctly parses the new delegating
   constructors.
*/
%module cpp11_delegating_constructors

%inline %{
class A {
public:
  int a;
  int b;
  int c;

  A() : A( 10 ) {}
  A(int aa) : A(aa, 20) {}
  A(int aa, int bb) : A(aa, bb, 30) {}
  A(int aa, int bb, int cc) { a=aa; b=bb; c=cc; }
};
%}
