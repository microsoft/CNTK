/* This testcase checks whether SWIG correctly parses the support for types
   without the defined trivial constructor in the unions. */
%module cpp11_unrestricted_unions

%inline %{
struct point {
  point() {}
  point(int x, int y) : x_(x), y_(y) {}
  int x_, y_;
};

#include <new> // For placement 'new' in the constructor below
union P {
  int z;
  double w;
  point p; // Illegal in C++03; legal in C++11.
  // Due to the point member, a constructor definition is required.
  P() {
    new(&p) point();
  }
} p1;
%}

