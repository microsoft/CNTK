%module(package="pkg1.pkg2") foo
%{
#include "../../pkg1/pkg2/foo.hpp"
%}
%import  "../../pkg1/foo.i"
%include "../../pkg1/pkg2/foo.hpp"
