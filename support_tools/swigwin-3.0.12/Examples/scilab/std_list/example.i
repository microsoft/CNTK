/* File : example.i */

%module example

%{
#include "example.h"
%}

%include stl.i
%include std_list.i

/* instantiate the required template specializations */
namespace std
{
    %template(IntList) list<int>;
    %template(StringList) list<std::string>;
}

%include "example.h"
