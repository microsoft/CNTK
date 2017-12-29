// Tests copy constructor
%module overload_copy

#ifndef SWIG_NO_OVERLOAD
%inline %{

class Foo {
public:
    Foo() { }
    Foo(const Foo &) { }
};

%}

#endif

