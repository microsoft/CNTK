%module(package="py2.pkg2") bar
%{
#include "../../py2/pkg2/bar.hpp"
%}
%import  "../../py2/pkg2/pkg3/foo.i"
%include "../../py2/pkg2/bar.hpp"
