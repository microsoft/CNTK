%module overload_return_type

// Regression test for PHP from SF#3168531 (SWIG <= 2.0.1 segfaults).

%inline %{

class A { };
class B {
    public:
        int foo(int x) { return 0; }
        A foo(const char * y) { return A(); }
};

// Regression test for PHP from SF#3208299 (there bar()'s return type wa
// treated as always void).

void foo(int i) {}
int foo() { return 1; }

int bar() { return 1; }
void bar(int i) {}

%}
