%module li_std_vector_member_var

%include "std_vector.i"

%template(vectorDbl) std::vector<double>;

%inline %{
#include <vector>

typedef std::vector<double> DblVector;

struct Test {
    DblVector v;
    int x;

    Test() : x(0) { }

    void f(int n) {
        x += n;
        v.push_back(1.0 / n);
    }
};

// Regression test for SF#3528035:
struct S {
    int x;
    S() : x(4) { }
};

struct T {
    S start_t;
    unsigned length;
};
%}
