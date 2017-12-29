%module(package="py2.pkg2") bar
%{
#include "../../py2/pkg2/bar.hpp"
%}
%import  "../../py2/pkg2/pkg3/pkg4/foo.i"
%include "../../py2/pkg2/bar.hpp"
