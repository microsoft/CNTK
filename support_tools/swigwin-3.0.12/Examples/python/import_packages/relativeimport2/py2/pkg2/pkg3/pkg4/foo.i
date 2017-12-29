%module(package="py2.pkg2.pkg3.pkg4") foo
%{
#include "../../../../py2/pkg2/pkg3/pkg4/foo.hpp"
%}
%include "../../../../py2/pkg2/pkg3/pkg4/foo.hpp"
