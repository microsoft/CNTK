
class A {
  public:
    void fA() {}
};

typedef A tA;

void test_A(A *a) {}
void test_tA(tA *a) {}

tA *new_tA() { return new tA(); }
