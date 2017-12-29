// Test ruby_minherit (multiple inheritance support) and shared_ptr
%module(ruby_minherit="1") ruby_minherit_shared_ptr

%include <boost_shared_ptr.i>
%shared_ptr(Interface1)
%shared_ptr(Base1)
%shared_ptr(MultiDerived)

%inline %{
#include <boost/shared_ptr.hpp>
class Interface1 {
public:
  virtual int Interface1Func() const = 0;
};

class Base1 {
  int val;
public:
  Base1(int a = 0) : val(a) {}
  virtual int Base1Func() const { return val; }
};

class MultiDerived : public Base1, public Interface1 {
  int multi;
public:
  MultiDerived(int v1, int v2) : Base1(v1), multi(v2) {}
  virtual int Interface1Func() const { return multi; }
};

int BaseCheck(const Base1& b) {
    return b.Base1Func();
}
int InterfaceCheck(const Interface1& i) {
    return i.Interface1Func();
}
int DerivedCheck(const MultiDerived& m) {
    return m.Interface1Func() + m.Base1Func();
}
%}
