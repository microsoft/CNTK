/* This testcase shows a few simple ways to deal with the new initializer_list
   introduced in C++11. */
%module cpp11_initializer_list

%warnfilter(SWIGWARN_TYPEMAP_INITIALIZER_LIST) B::B;
%ignore A::A(std::initializer_list<int>);
%ignore B::method;

%typemap(in) std::initializer_list<const char *> {
  $1 = {"Ab", "Fab"};
}

%inline %{
#include <initializer_list>

class A {
public:
  A(std::initializer_list<int>) {}
  A() {}
  A(double d) {}
};
class B {
public:
  B(std::initializer_list<int>, std::initializer_list<double>) {}
  B() {}
  void method(std::initializer_list<int> init) {}
};
class C {
public:
  C(std::initializer_list<const char *>) {}
  C() {}
};
%}

