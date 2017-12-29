/* This testcase checks whether SWIG correctly treats the new nullptr_t
   constant introduced in C++11.
*/

%module cpp11_null_pointer_constant

%feature("autodoc") A::NullPtrMethod; // Triggers conversion of nullptr to None, nil etc in target language
%feature("compactdefaultargs") A::NullPtrMethod;

%inline %{

const int *const MyIntegerPtr = nullptr;

class A {
public:
  A() : _myA(nullptr) { }

  A *_myA;
  
  void NullPtrMethod(double *ptr = nullptr) {}  
};
%}
