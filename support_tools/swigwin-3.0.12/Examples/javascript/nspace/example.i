%module example

%{
#include "example.h"
%}

%nspace MyWorld::Nested::Dweller;
%nspace MyWorld::World;

%include "example.h"
