%module mod_a

%{
#include "mod.h"
%}


class C;

class A
{
public:
    A() {}
    C* GetC() { return NULL; }

    void DoSomething(A* a) {}
};


class B : public A
{
public:
    B();
};
