// Tests classes passed by value, pointer and reference
// Note: C# module has a large runtime test

%warnfilter(SWIGWARN_TYPEMAP_THREAD_UNSAFE,SWIGWARN_TYPEMAP_DIRECTOROUT_PTR) Base::Ref;
%warnfilter(SWIGWARN_TYPEMAP_THREAD_UNSAFE,SWIGWARN_TYPEMAP_DIRECTOROUT_PTR) Base::Ptr;

%module(directors="1") director_classes

%feature("director") Base;
%feature("director") Derived;

%include "std_string.i"

%{
#if defined(__SUNPRO_CC)
#pragma error_messages (off, hidevf)
#endif
%}

%inline %{
#include <cstdio>
#include <iostream>


// Use for debugging
bool PrintDebug = false;


struct DoubleHolder
{
  DoubleHolder(double v = 0.0) : val(v) {}
  double val;
};

class Base {
protected:
  double m_dd;
public:

  Base(double dd) : m_dd(dd) {}
  virtual ~Base() {}

  virtual DoubleHolder Val(DoubleHolder x) { if (PrintDebug) std::cout << "Base - Val(" << x.val << ")" << std::endl; return x; }
  virtual DoubleHolder& Ref(DoubleHolder& x) { if (PrintDebug) std::cout << "Base - Ref(" << x.val << ")" << std::endl; return x; }
  virtual DoubleHolder* Ptr(DoubleHolder* x) { if (PrintDebug) std::cout << "Base - Ptr(" << x->val << ")" << std::endl; return x; }

  virtual std::string FullyOverloaded(int x) { if (PrintDebug) std::cout << "Base - FullyOverloaded(int " << x << ")" << std::endl; return "Base::FullyOverloaded(int)"; }
  virtual std::string FullyOverloaded(bool x) { if (PrintDebug) std::cout << "Base - FullyOverloaded(bool " << x << ")" << std::endl; return "Base::FullyOverloaded(bool)"; }

  virtual std::string SemiOverloaded(int x) { if (PrintDebug) std::cout << "Base - SemiOverloaded(int " << x << ")" << std::endl; return "Base::SemiOverloaded(int)"; }
  virtual std::string SemiOverloaded(bool x) { if (PrintDebug) std::cout << "Base - SemiOverloaded(bool " << x << ")" << std::endl; return "Base::SemiOverloaded(bool)"; }

  virtual std::string DefaultParms(int x, double y = 1.1) {
    if (PrintDebug) std::cout << "Base - DefaultParms(" << x << ", " << y << ")" << std::endl;
    std::string ret("Base::DefaultParms(int");
    if (y!=1.1)
      ret = ret + std::string(", double");
    ret = ret + std::string(")");
    return ret;
  }
};

class Derived : public Base {
public:
  Derived(double dd) : Base(dd) {}
  virtual ~Derived() {}

  virtual DoubleHolder Val(DoubleHolder x) { if (PrintDebug) std::cout << "Derived - Val(" << x.val << ")" << std::endl; return x; }
  virtual DoubleHolder& Ref(DoubleHolder& x) { if (PrintDebug) std::cout << "Derived - Ref(" << x.val << ")" << std::endl; return x; }
  virtual DoubleHolder* Ptr(DoubleHolder* x) { if (PrintDebug) std::cout << "Derived - Ptr(" << x->val << ")" << std::endl; return x; }

  virtual std::string FullyOverloaded(int x) { if (PrintDebug) std::cout << "Derived - FullyOverloaded(int " << x << ")" << std::endl; return "Derived::FullyOverloaded(int)"; }
  virtual std::string FullyOverloaded(bool x) { if (PrintDebug) std::cout << "Derived - FullyOverloaded(bool " << x << ")" << std::endl; return "Derived::FullyOverloaded(bool)"; }

  virtual std::string SemiOverloaded(int x) { if (PrintDebug) std::cout << "Derived - SemiOverloaded(int " << x << ")" << std::endl; return "Derived::SemiOverloaded(int)"; }
  // No SemiOverloaded(bool x)

  virtual std::string DefaultParms(int x, double y = 1.1) { 
    if (PrintDebug) std::cout << "Derived - DefaultParms(" << x << ", " << y << ")" << std::endl;
    std::string ret("Derived::DefaultParms(int");
    if (y!=1.1)
      ret = ret + std::string(", double");
    ret = ret + std::string(")");
    return ret;
  }
};


class Caller {
private:
  Base *m_base;
  void delBase() { delete m_base; m_base = 0; }
public:
  Caller(): m_base(0) {}
  ~Caller() { delBase(); }
  void set(Base *b) { delBase(); m_base = b; }
  void reset() { m_base = 0; }

  DoubleHolder ValCall(DoubleHolder x) { return m_base->Val(x); }
  DoubleHolder& RefCall(DoubleHolder& x) { return m_base->Ref(x); }
  DoubleHolder* PtrCall(DoubleHolder* x) { return m_base->Ptr(x); }
  std::string FullyOverloadedCall(int x) { return m_base->FullyOverloaded(x); }
  std::string FullyOverloadedCall(bool x) { return m_base->FullyOverloaded(x); }
  std::string SemiOverloadedCall(int x) { return m_base->SemiOverloaded(x); }
  std::string SemiOverloadedCall(bool x) { return m_base->SemiOverloaded(x); }
  std::string DefaultParmsCall(int x) { return m_base->DefaultParms(x); }
  std::string DefaultParmsCall(int x, double y) { return m_base->DefaultParms(x, y); }
};

%}


%feature(director) BaseClass;
%feature(director) DerivedClass;

%inline %{
class BaseClass
{
public:
virtual ~BaseClass() {};
virtual int dofoo(int& one, int& two, int& three) {return 0;}
};

class DerivedClass : public BaseClass
{
};
%}

