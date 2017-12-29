%module(package="pkg1") foo
%{
#include "../pkg1/foo.hpp"
%}
%include "../pkg1/foo.hpp"
