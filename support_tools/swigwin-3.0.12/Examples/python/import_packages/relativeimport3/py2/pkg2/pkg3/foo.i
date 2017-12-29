%module(package="py2.pkg2.pkg3") foo
%{
#include "../../../py2/pkg2/pkg3/foo.hpp"
%}
%include "../../../py2/pkg2/pkg3/foo.hpp"
