/* This testcase checks whether SWIG correctly uses the sizeof() on the
   concrete objects and not only types introduced in C++11. */
%module cpp11_sizeof_object

%inline %{
struct B {
  unsigned long member1;
  long long member2;
  char member3;
};

struct A {
  B member;
};

const int a = sizeof(A::member);
%}
