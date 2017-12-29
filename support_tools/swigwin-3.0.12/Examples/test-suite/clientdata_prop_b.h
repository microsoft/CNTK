#include "clientdata_prop_a.h"

typedef tA t2A;
typedef A t3A;

class B : public A
{
  public:
    void fB() {}
};

class C : public tA
{
  public:
    void fC() {}
};

class D : public t2A
{
  public:
    void fD() {}
};

typedef D tD;
typedef tD t2D;

void test_t2A(t2A *a) {}
void test_t3A(t3A *a) {}
void test_B(B *b) {}
void test_C(C *c) {}
void test_D(D *d) {}
void test_tD(tD *d) {}
void test_t2D(t2D *d) {}

t2A *new_t2A() { return new t2A(); }
t3A *new_t3A() { return new t3A(); }
tD * new_tD () { return new tD (); }
t2D *new_t2D() { return new t2D(); }
