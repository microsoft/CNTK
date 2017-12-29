%module mod_b

%{
#include "mod.h"
%}


%import mod_a.i


class C : public B
{
public:
    C() {}
};


class D : public C
{
public:
    D() {}
};
