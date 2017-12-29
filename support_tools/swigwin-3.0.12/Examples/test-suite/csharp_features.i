%module csharp_features
%include "wchar.i"

// SWIG gets the method modifiers wrong occasionally, like with private inheritance, %csmethodmodifiers can fix this
%csmethodmodifiers Derived::VirtualMethod() "public virtual"
%csmethodmodifiers MoreDerived::variable "public new"

%inline %{
class Base {
public:
  virtual ~Base() {}
  virtual void VirtualMethod() {}
};
class Derived : private Base {
public:
  virtual ~Derived() {}
  virtual void VirtualMethod() {}
  int variable;
};
class MoreDerived : public Derived {
public:
  int variable;
  // test wide char literals support for C# module
  void methodWithDefault1(const wchar_t* s = L"literal with escapes \x1234"){}
  void methodWithDefault2(wchar_t c = L'\x1234'){}
};
%}

