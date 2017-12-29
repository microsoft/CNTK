%module(package="PackageC") "packageoption_c";

%import "packageoption_a.i"

%inline %{
#include "packageoption.h"

struct Derived : Base {
  virtual int vmethod() { return 2; }
  virtual ~Derived() {}
};

%}
