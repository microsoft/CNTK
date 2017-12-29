%module base
%{
#include "base.h"
%}

%include base.h
%template(IntBase) Base<int>;
