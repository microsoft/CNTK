%module example

%{
#include "example.h"
%}

/**
 * These overloaded declarations conflict with other overloads (as far as
 * SWIG's Ruby module's implementation for overloaded methods is concerned).
 * One option is use the %rename directive to rename the conflicting methods;
 * here, we're just using %ignore to avoid wrapping some of the overloaded
 * functions altogether.
 */

%ignore Bar::Bar(Bar *);
%ignore Bar::Bar(long);

%ignore Bar::foo(const Bar&);
%ignore Bar::foo(long);

%ignore ::foo(const Bar&);
%ignore ::foo(int);

%include example.h
