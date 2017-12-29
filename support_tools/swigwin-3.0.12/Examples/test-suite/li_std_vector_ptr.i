// SF Bug 2359417
%module li_std_vector_ptr

%include "std_vector.i"

%template(IntPtrVector) std::vector<int *>;

%inline %{
#include <iostream>
using namespace std;
int* makeIntPtr(int v) {
  return new int(v);
}
double* makeDoublePtr(double v) {
  return new double(v);
}

// pointer to pointer in the wrappers was preventing a vector of pointers from working
int** makeIntPtrPtr(int* v) {
  return new int*(v);
}

void displayVector(std::vector<int *> vpi) {
  cout << "displayVector..." << endl;
  for (size_t i=0; i<vpi.size(); ++i)
    cout << *vpi[i] << endl;
}
int getValueFromVector(std::vector<int *> vpi, size_t index) {
  return *vpi[index];
}
%}

// A not exposed to wrappers
%{
struct A {
  int val;
  A(int val) : val(val) {}
};
%}

%template(APtrVector) std::vector<A *>;

%inline %{
A *makeA(int val) { return new A(val); }
int getVal(A* a) { return a->val; }
int getVectorValueA(std::vector<A *> vpi, size_t index) {
  return vpi[index]->val;
}
%}

// B is fully exposed to wrappers
%inline %{
struct B {
  int val;
  B(int val = 0) : val(val) {}
};
%}

%template(BPtrVector) std::vector<B *>;

%inline %{
B *makeB(int val) { return new B(val); }
int getVal(B* b) { return b->val; }
int getVectorValueB(std::vector<B *> vpi, size_t index) {
  return vpi[index]->val;
}
%}

// C is fully exposed to wrappers (includes code using B **)
%inline %{
struct C {
  int val;
  C(int val = 0) : val(val) {}
};
%}

%template(CPtrVector) std::vector<C *>;

%inline %{
// pointer to pointer in the wrappers was preventing a vector of pointers from working
C** makeCIntPtrPtr(C* v) {
  return new C*(v);
}
C *makeC(int val) { return new C(val); }
int getVal(C* b) { return b->val; }
int getVectorValueC(std::vector<C *> vpi, size_t index) {
  return vpi[index]->val;
}
%}

