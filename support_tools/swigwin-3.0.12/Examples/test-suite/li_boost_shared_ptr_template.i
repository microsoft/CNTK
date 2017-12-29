%module li_boost_shared_ptr_template

// First test- Bug 3333549 - using INTEGER typedef in %shared_ptr before typedef defined
%{
#include <boost/shared_ptr.hpp>

#ifdef SWIGR
  // remove naming conflict with R symbol
#define INTEGER MYINTEGER
#endif

  typedef int INTEGER;

  template <class T> 
    class Base {
  public:
    virtual T bar() {return 1;}
    virtual ~Base() {}
  };

  template <class T> 
    class Derived : public Base<T> {
  public:
    virtual T bar() {return 2;}
  };
  
  INTEGER bar_getter(Base<INTEGER>& foo) {
    return foo.bar();
  }

%}

#if defined(SWIGJAVA) || defined(SWIGCSHARP) || defined(SWIGPYTHON) || defined(SWIGD) || defined(SWIGOCTAVE) || defined(SWIGRUBY)
#define SHARED_PTR_WRAPPERS_IMPLEMENTED
#endif

#if defined(SHARED_PTR_WRAPPERS_IMPLEMENTED)

%include <boost_shared_ptr.i>
%shared_ptr(Base<INTEGER>)
%shared_ptr(Derived<INTEGER>)

#endif

typedef int INTEGER;

template <class T> 
class Base {
  public:
  virtual T bar() {return 1;}
};

template <class T> 
class Derived : public Base<T> {
  public:
  virtual T bar() {return 2;}
};

INTEGER bar_getter(Base<INTEGER>& foo) {
  return foo.bar();
}

%template(BaseINTEGER) Base<INTEGER>;
%template(DerivedINTEGER) Derived<INTEGER>;
 

// 2nd test - templates with default template parameters
#if defined(SHARED_PTR_WRAPPERS_IMPLEMENTED)

%shared_ptr(Space::BaseDefault<short, int>)
%shared_ptr(Space::DerivedDefault<short>)
%shared_ptr(Space::DerivedDefault2<short>)

#endif

%inline %{
namespace Space {
typedef int INT_TYPEDEF;
template <class X, class T = int> 
class BaseDefault {
  public:
  virtual T bar2() {return 3;}
  virtual ~BaseDefault() {}
};

template <class X, class T = int> 
class DerivedDefault : public BaseDefault<X, T> {
  public:
  virtual T bar2() {return 4;}
};
template <class X> 
class DerivedDefault2 : public BaseDefault<X> {
  public:
  virtual int bar2() {return 4;}
};

int bar2_getter(BaseDefault<short>& foo) {
  return foo.bar2();
}
}
%}

%template(BaseDefaultInt) Space::BaseDefault<short>;
%template(DerivedDefaultInt) Space::DerivedDefault<short>;
%template(DerivedDefaultInt2) Space::DerivedDefault2<short>;
 
