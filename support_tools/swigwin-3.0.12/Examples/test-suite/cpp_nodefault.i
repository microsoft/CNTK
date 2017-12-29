// This file tests SWIG pass/return by value for
// a class with no default constructor

%module cpp_nodefault

%inline %{

class Foo {
public:
   int a;
   Foo(int x, int y) { }
  ~Foo() {}
};

Foo create(int x, int y) {
    return Foo(x,y);
}

typedef Foo Foo_t;

void consume(Foo f, Foo_t g) {}

class Bar {
public:
    void consume(Foo f, Foo_t g) {}
    Foo create(int x, int y) {
        return Foo(x,y);
    }
};


%}

%{
Foo gvar = Foo(3,4);
%}

Foo gvar;


