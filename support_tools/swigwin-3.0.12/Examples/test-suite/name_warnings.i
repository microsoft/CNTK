%module name_warnings
/*
  This test should produce no warnings at all.

  It just show the cases where swig was showing unintended warnings
  before.

  Right now the test includes some cases for python, java and csharp.

*/

/* activate all the name warnings */
%warnfilter(+SWIGWARN_PARSE_KEYWORD,+SWIGWARN_PARSE_BUILTIN_NAME,-SWIGWARN_TYPE_ABSTRACT);

%{
#ifdef max
#undef max
#endif
%}

class string; // csharp keyword
namespace std 
{
  template <class T>
    class complex;
}

%inline 
{
  class complex; // python built-in

  typedef complex None;  // python built-in
  
  struct A 
  {
    typedef complex None;
    
#ifndef SWIGPHP // clone() *is* an invalid method name in PHP.
    A* clone(int) { return NULL; }
#endif
    
    virtual ~A() {}
#ifndef SWIGGO // func is a keyword in Go.
    virtual int func() = 0;
#endif
  private:
     typedef complex False;
  };

  template <class T>
    T max (T a, T b) { // python 'max' built-in
    return a > b ? a : b;
  }  

  struct B : A
  {
    B() {}
  };
  
  
}

%template(max_i) max<int>;

%inline {
  /* silently rename the parameter names in csharp/java */
#ifdef SWIGR
  double foo(double inparam, double out) { return 1.0; }
#else
  double foo(double abstract, double out) { return 1.0; }
#endif
  double bar(double native, bool boolean) { return 1.0; }
}
