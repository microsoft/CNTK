%module(package="CommonPackage") "packageoption_a";

%inline %{
class A
{
 public:
  int testInt() { return 2;}
};
%}

%{
#include "packageoption.h"
%}

%include "packageoption.h"

