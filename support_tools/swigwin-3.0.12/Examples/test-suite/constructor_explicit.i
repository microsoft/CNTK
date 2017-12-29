/* Swig 1.3.6 fails to understand the "explicit" keyword.
   SF Bug #445233, reported by Krzysztof Kozminski
   <kozminski@users.sf.net>. 
*/

%module constructor_explicit
%inline %{

class Foo {
public:
   explicit Foo() { }
   explicit Foo(int) {};
};

Foo test(Foo x) {
   return x;
}

%}
