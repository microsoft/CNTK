#if !defined(SWIGPYTHON_BUILTIN)
%define MODULEIMPORT
"
# print 'Loading low-level module $module'
import $module
# print 'Module has loaded'
extra_import_variable = 'custom import of $module'
"
%enddef

#else
%define MODULEIMPORT
"
# print 'Loading low-level module $module'
extra_import_variable = 'custom import of $module'
from $module import *
# print 'Module has loaded'
"
%enddef
#endif

%module(moduleimport=MODULEIMPORT) python_moduleimport

%inline %{
int simple_function(int i) { return i; }
%}
