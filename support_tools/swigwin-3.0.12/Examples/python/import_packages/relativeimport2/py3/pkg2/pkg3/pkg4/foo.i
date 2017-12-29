%module(package="py3.pkg2.pkg3.pkg4") foo
%{
#include "../../../../py3/pkg2/pkg3/pkg4/foo.hpp"
%}
%include "../../../../py3/pkg2/pkg3/pkg4/foo.hpp"
