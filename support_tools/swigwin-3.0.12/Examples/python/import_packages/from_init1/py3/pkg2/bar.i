%module(package="py3.pkg2") bar
%{
#include "../../py3/pkg2/bar.hpp"
%}
%import  "../../py3/pkg2/foo.i"
%include "../../py3/pkg2/bar.hpp"
