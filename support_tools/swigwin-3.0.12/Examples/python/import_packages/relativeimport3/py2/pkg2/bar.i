%module(package="py2.pkg2") bar
%{
#include "../../py2/pkg2/bar.hpp"
%}
%import (module="foo", package="py2.pkg2.pkg3") "../../py2/pkg2/pkg3/foo.hpp"
%include "../../py2/pkg2/bar.hpp"
