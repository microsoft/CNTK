%module(package="pkg2") foo
%{
#include "../pkg2/foo.hpp"
%}
%import  "../pkg1/foo.i"
%include "../pkg2/foo.hpp"
