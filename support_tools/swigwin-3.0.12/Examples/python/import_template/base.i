%module base
%{
#include "base.h"
%}

%include base.h
%template(intBase) Base<int>;
