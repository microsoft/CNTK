%module(package="py3.pkg2") foo
%{
#include "../../py3/pkg2/foo.hpp"
%}
%include "../../py3/pkg2/foo.hpp"
