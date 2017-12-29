%module li_swigtype_inout

// Test SWIGTYPE *& typemaps in swigtype_inout.i library

#ifdef SWIGCSHARP
%include <swigtype_inout.i>
%apply SWIGTYPE *& OUTPUT { SWIGTYPE *& }
#endif

%ignore XXX::operator=;

%inline %{
#include <iostream>
struct XXX {
  XXX(int v) : value(v) {
    if (debugging) std::cout << "Default Constructor " << value << " " << this << std::endl;
    count++;
  }
  XXX(const XXX &other) {
    value = other.value;
    if (debugging) std::cout << "Copy    Constructor " << value << " " << this << std::endl;
    count++;
  }
  XXX& operator=(const XXX &other) {
    value = other.value;
    if (debugging) std::cout << "Assignment operator " << value << " " << this << std::endl;
    return *this;
  }
  ~XXX() {
    if (debugging) std::cout << "Destructor          " << value << " " << this << std::endl;
    count--;
  }
  void showInfo() {
    if (debugging) std::cout << "Info                " << value << " " << this << std::endl;
  }
  int value;
  static const bool debugging = false;
  static int count;
};
int XXX::count = 0;

void ptr_ref_out(XXX *& x1, XXX *& x2, XXX const*& x3, XXX const*& x4) {
  x1 = new XXX(111);
  x2 = new XXX(222);
  x3 = new XXX(333);
  x4 = new XXX(444);
}
struct ConstructorTest {
  ConstructorTest(XXX *& x1, XXX *& x2, XXX const*& x3, XXX const*& x4) {
    ptr_ref_out(x1, x2, x3, x4);
  }
};
%}
