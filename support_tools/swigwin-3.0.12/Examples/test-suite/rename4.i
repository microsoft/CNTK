// Test %rename directive with the 'using' keyword and within the class definition
%module rename4

%{
#include "rename.h"
%}

namespace Space {
struct Klass {
  Klass(int i) {}
  Klass() {}
};
}

namespace AnotherSpace {
  class Another {};
}

namespace Space {
  %rename(opAnother1) XYZ::operator Another() const;
  %rename(opAnother2) XYZ<int>::operator Another() const;
  %rename(opAnother3) XYZ<Space::Klass>::operator Another() const;
  %rename(opAnother4) XYZ<Space::Enu>::operator Another() const;
}

// Test %rename - no namespace, but specific templated type in the parameter, is used over the generic type T
%rename(tMethod2) templateT(int i);
%rename(tMethodNotXYZ2) templateNotXYZ(NotXYZ<int>);
%rename(tMethodXYZ2) templateXYZ(XYZ<int>);
%rename(opT2) operator int();
%rename(opNotXYZ2) operator NotXYZ<int>() const;
%rename(opXYZ2) operator XYZ<int>() const;

%rename(tMethod3) templateT(Space::Klass i);
%rename(tMethodNotXYZ3) templateNotXYZ(NotXYZ<Space::Klass>);
%rename(tMethodXYZ3) templateXYZ(XYZ<Space::Klass>);
%rename(opT3) operator Space::Klass();
%rename(opNotXYZ3) operator NotXYZ<Space::Klass>() const;
%rename(opXYZ3) operator XYZ<Space::Klass>() const;

%rename(tMethod4) templateT(Space::Enu i);
%rename(tMethodNotXYZ4) templateNotXYZ(NotXYZ<Space::Enu>);
%rename(tMethodXYZ4) templateXYZ(XYZ<Space::Enu>);
%rename(opT4) operator Space::Enu();
%rename(opNotXYZ4) operator NotXYZ<Space::Enu>() const;
%rename(opXYZ4) operator XYZ<Space::Enu>() const;

namespace Space {
  using namespace AnotherSpace;
  enum Enu { En1, En2, En3 };
  template<typename T> struct NotXYZ {};
  template<typename T> class XYZ {

    // Test %rename within the class
    %rename(opIntPtrA) operator NotXYZ<int>*() const;
    %rename(opIntPtrB) operator XYZ<int>*() const;

    %rename(tMethod1) templateT(T i);
    %rename(tMethodNotXYZ1) templateNotXYZ(NotXYZ<T>);
    %rename(tMethodXYZ1) templateXYZ(XYZ<T>);
    %rename(opT1) operator T();
    %rename(opNotXYZ1) operator NotXYZ<T>() const;
    %rename(opXYZ1) operator XYZ<T>() const;

    NotXYZ<int> *m_int;
    T m_t;
    NotXYZ<T> m_notxyz;
  public:
    operator NotXYZ<int>*() const { return m_int; }
    operator XYZ<int>*() const { return 0; }
    operator Another() const { Another an; return an; }
    void templateT(T i) {}
    void templateNotXYZ(NotXYZ<T> i) {}
    void templateXYZ(XYZ<T> i) {}
    operator T() { return m_t; }
    operator NotXYZ<T>() const { return m_notxyz; }
    operator XYZ<T>() const { XYZ<T> xyz; return xyz; }
  };
}

%exception Space::ABC::operator ABC %{
#if defined(__clang__)
  // Workaround for: warning: conversion function converting 'Space::ABC' to itself will never be used
  result = *arg1;
#else
  $action
#endif
%}

namespace Space {
// non-templated class using itself in method and operator
class ABC {
  public:

    %rename(methodABC) method(ABC a) const;
    %rename(opABC) operator ABC() const;
    %rename(methodKlass) method(Klass k) const;
    %rename(opKlass) operator Klass() const;

    void method(ABC a) const {}
    void method(Klass k) const {}
#if !defined(__clang__)
    // Workaround for: warning: conversion function converting 'Space::ABC' to itself will never be used
    operator ABC() const { ABC a; return a; }
#endif
    operator Klass() const { Klass k; return k; }
};
}


%template(XYZInt) Space::XYZ<int>;
%template(XYZDouble) Space::XYZ<double>;
%template(XYZKlass) Space::XYZ<Space::Klass>;
%template(XYZEnu) Space::XYZ<Space::Enu>;

%template(NotXYZInt) Space::NotXYZ<int>;
%template(NotXYZDouble) Space::NotXYZ<double>;
%template(NotXYZKlass) Space::NotXYZ<Space::Klass>;
%template(NotXYZEnu) Space::NotXYZ<Space::Enu>;

