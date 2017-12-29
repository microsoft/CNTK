%module cpp11_reference_wrapper

// SWIG could provide some sort of typemaps for reference_wrapper which is acts like a C++ reference,
// but is copy-constructible and copy-assignable

%inline %{
#include <iostream>
#include <functional>
using namespace std;

struct B {
  B(int &val) : val(val) {}
  std::reference_wrapper<int> val;
//  int &val;
};
%}

%inline %{
void go() {
  int val(999);
  B b1(val);
  int const& aa1 = b1.val;
  cout << aa1 << endl;

  // copy constructible
  B b2(b1);
  int const& aa2 = b2.val;
  cout << aa2 << endl;

  // copy assignable
  B b3(val);
  b3 = b1;
  int const& aa3 = b3.val;
  cout << aa3 << endl;
}
%}
