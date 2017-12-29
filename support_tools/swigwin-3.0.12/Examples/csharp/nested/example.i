%module example

// This example shows how wrappers for numerous aspects of C++ nested classes work:
// Nested static and instance variables and methods and nested enums

%include <std_string.i>

%{
#include "example.h"
%}

%include "example.h"

