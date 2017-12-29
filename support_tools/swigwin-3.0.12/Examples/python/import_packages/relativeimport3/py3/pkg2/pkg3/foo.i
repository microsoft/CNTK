%module(package="py3.pkg2.pkg3") foo
%{
#include "../../../py3/pkg2/pkg3/foo.hpp"
%}
%include "../../../py3/pkg2/pkg3/foo.hpp"
