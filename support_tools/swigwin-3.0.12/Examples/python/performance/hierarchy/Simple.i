%inline %{

class A {
public:
    A () {}
    ~A () {}
    void func () {}
};

class B : public A {
public:
    B () {}
    ~B () {}
};

class C : public B {
public:
    C () {}
    ~C () {}
};

class D : public C {
public:
    D () {}
    ~D () {}
};

class E : public D {
public:
    E () {}
    ~E () {}
};

class F : public E {
public:
    F () {}
    ~F () {}
};

class G : public F {
public:
    G () {}
    ~G () {}
};

class H : public G {
public:
    H () {}
    ~H () {}
};

%}
