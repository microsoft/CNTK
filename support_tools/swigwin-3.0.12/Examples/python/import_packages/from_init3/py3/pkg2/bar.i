%module(package="py3.pkg2") bar
%{
#include "../../py3/pkg2/bar.hpp"
%}
%import  "../../py3/pkg2/pkg3/pkg4/foo.i"
%include "../../py3/pkg2/bar.hpp"
