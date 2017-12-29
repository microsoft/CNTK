%module cpp_shared_ptr

%include <boost_shared_ptr.i>

%shared_ptr(B);
%shared_ptr(C);

%inline %{
  #include <stdio.h>
  #include <boost/shared_ptr.hpp>

  struct A {
    virtual ~A() {}
  };

  struct B {
    virtual ~B() {}
  };

  struct C : B, A {
    virtual ~C() {}
  };

  struct D : C {
    virtual ~D() {}
  };
%}


