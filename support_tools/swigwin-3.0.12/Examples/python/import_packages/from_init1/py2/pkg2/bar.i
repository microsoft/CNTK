%module(package="py2.pkg2") bar
%{
#include "../../py2/pkg2/bar.hpp"
%}
%import  "../../py2/pkg2/foo.i"
%include "../../py2/pkg2/bar.hpp"
