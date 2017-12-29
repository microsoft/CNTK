%module example
%include <cni.i>

%{
#include "Example.h"
%}


%include Example.h

%extend Example {
  ~Example() {}
}
