%module(package="py2.pkg2") foo
%{
#include "../../py2/pkg2/foo.hpp"
%}
%include "../../py2/pkg2/foo.hpp"
