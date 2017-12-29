%module inline_initializer

%inline %{
class Foo {
   int x;
public:
   Foo(int a);
};

Foo::Foo(int a) : x(a) { }

%}
