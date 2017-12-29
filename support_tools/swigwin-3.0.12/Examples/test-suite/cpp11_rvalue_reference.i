/* This testcase checks whether SWIG correctly parses the double ampersand &&
   move operator which is currently mapped to the reference & operator. */
%module cpp11_rvalue_reference

%inline %{
#include <utility>
class A {
public:
  int  getAcopy() { return _a; }
  int *getAptr()  { return &_a; }
  int &getAref()  { return _a; }
  int &&getAmove() { return std::move(_a); }

  void setAcopy(int a) { _a = a; }
  void setAptr(int *a)  { _a = *a; }
  void setAref(int &a)  { _a = a; }
  void setAmove(int &&a) { _a = a; }

private:
  int _a;
};
%}
