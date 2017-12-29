// Tests primitives
// Note: C# module has a large runtime test

#pragma SWIG nowarn=SWIGWARN_TYPEMAP_THREAD_UNSAFE,SWIGWARN_TYPEMAP_DIRECTOROUT_PTR

%module(directors="1") director_primitives

%feature("director") Base;
%feature("director") Derived;

%include "std_string.i"

%inline %{
#include <cstdio>
#include <iostream>


// Use for debugging
bool PrintDebug = false;

enum HShadowMode
{
  HShadowNone = 1,
  HShadowSoft = 2,
  HShadowHard = 3
};

class Base {
protected:
  double m_dd;
public:

  Base(double dd) : m_dd(dd) {}
  virtual ~Base() {}

  virtual void NoParmsMethod() { if (PrintDebug) std::cout << "Base - NoParmsMethod()" << std::endl; }
  virtual bool BoolMethod(bool x) { if (PrintDebug) std::cout << "Base - BoolMethod(" << x << ")" << std::endl; return x; }
  virtual int IntMethod(int x) { if (PrintDebug) std::cout << "Base - IntMethod(" << x << ")" << std::endl; return x; }
  virtual unsigned int UIntMethod(unsigned int x) { if (PrintDebug) std::cout << "Base - UIntMethod(" << x << ")" << std::endl; return x; }
  virtual float FloatMethod(float x) { if (PrintDebug) std::cout << "Base - FloatMethod(" << x << ")" << std::endl; return x; }
  virtual char * CharPtrMethod(char * x) { if (PrintDebug) std::cout << "Base - CharPtrMethod(" << x << ")" << std::endl; return x; }
  virtual const char * ConstCharPtrMethod(const char * x) { if (PrintDebug) std::cout << "Base - ConstCharPtrMethod(" << x << ")" << std::endl; return x; }
  virtual HShadowMode EnumMethod(HShadowMode x) { if (PrintDebug) std::cout << "Base - EnumMethod(" << x << ")" << std::endl; return x; }
  virtual void ManyParmsMethod(bool b, int i, unsigned int u, float f, char * c, const char * cc, HShadowMode h) { if (PrintDebug) std::cout << "Base - ManyParmsMethod(" << b << ", " << i << ", " << u << ", " << f << ", " << c << ", " << cc << ", " << h << ")" << std::endl; }
  virtual void NotOverriddenMethod() { if (PrintDebug) std::cout << "Base - NotOverriddenMethod()" << std::endl; }
};

class Derived : public Base {
public:
  Derived(double dd) : Base(dd) {}
  virtual ~Derived() {}

  virtual void NoParmsMethod() { if (PrintDebug) std::cout << "Derived - NoParmsMethod()" << std::endl; }
  virtual bool BoolMethod(bool x) { if (PrintDebug) std::cout << "Derived - BoolMethod(" << x << ")" << std::endl; return x; }
  virtual int IntMethod(int x) { if (PrintDebug) std::cout << "Derived - IntMethod(" << x << ")" << std::endl; return x; }
  virtual unsigned int UIntMethod(unsigned int x) { if (PrintDebug) std::cout << "Derived - UIntMethod(" << x << ")" << std::endl; return x; }
  virtual float FloatMethod(float x) { if (PrintDebug) std::cout << "Derived - FloatMethod(" << x << ")" << std::endl; return x; }
  virtual char * CharPtrMethod(char * x) { if (PrintDebug) std::cout << "Derived - CharPtrMethod(" << x << ")" << std::endl; return x; }
  virtual const char * ConstCharPtrMethod(const char * x) { if (PrintDebug) std::cout << "Derived - ConstCharPtrMethod(" << x << ")" << std::endl; return x; }
  virtual HShadowMode EnumMethod(HShadowMode x) { if (PrintDebug) std::cout << "Derived - EnumMethod(" << x << ")" << std::endl; return x; }
  virtual void ManyParmsMethod(bool b, int i, unsigned int u, float f, char * c, const char * cc, HShadowMode h) { if (PrintDebug) std::cout << "Derived - ManyParmsMethod(" << b << ", " << i << ", " << u << ", " << f << ", " << c << ", " << cc << ", " << h << ")" << std::endl; }
};


class Caller {
private:
  Base *m_base;
  void delBase() { delete m_base; m_base = 0; }
public:
  Caller(): m_base(0) {}
  virtual ~Caller() { delBase(); }
  void set(Base *b) { delBase(); m_base = b; }
  void reset() { m_base = 0; }

  void NoParmsMethodCall() { m_base->NoParmsMethod(); }
  bool BoolMethodCall(bool x) { return m_base->BoolMethod(x); }
  int IntMethodCall(int x) { return m_base->IntMethod(x); }
  unsigned int UIntMethodCall(unsigned int x) { return m_base->UIntMethod(x); }
  float FloatMethodCall(float x) { return m_base->FloatMethod(x); }
  char * CharPtrMethodCall(char * x) { return m_base->CharPtrMethod(x); }
  const char * ConstCharPtrMethodCall(const char * x) { return m_base->ConstCharPtrMethod(x); }
  HShadowMode EnumMethodCall(HShadowMode x) { return m_base->EnumMethod(x); }
  virtual void ManyParmsMethodCall(bool b, int i, unsigned int u, float f, char * c, const char * cc, HShadowMode h) { return m_base->ManyParmsMethod(b, i, u, f, c, cc, h); }
  virtual void NotOverriddenMethodCall() { m_base->NotOverriddenMethod(); }
};

%}

