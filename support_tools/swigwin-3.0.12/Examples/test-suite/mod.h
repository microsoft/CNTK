

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
  B() {}
  
};


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
