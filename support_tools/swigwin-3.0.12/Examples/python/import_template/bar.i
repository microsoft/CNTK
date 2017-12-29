%module bar
%{
#include "bar.h"
%}

%import base.i
%include "bar.h"

%template(intBar) Bar<int>;


