%module foo
%{
#include "foo.h"
%}

%import base.i
%include "foo.h"

%template(IntFoo) Foo<int>;

