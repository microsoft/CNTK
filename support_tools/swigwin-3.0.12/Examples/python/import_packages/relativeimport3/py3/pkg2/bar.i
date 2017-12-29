%module(package="py3.pkg2") bar
%{
#include "../../py3/pkg2/bar.hpp"
%}
%import (module="foo", package="py3.pkg2.pkg3") "../../py3/pkg2/pkg3/foo.hpp"
%include "../../py3/pkg2/bar.hpp"
