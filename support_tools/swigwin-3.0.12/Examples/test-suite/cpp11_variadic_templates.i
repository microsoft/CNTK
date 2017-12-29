/* This testcase checks whether SWIG correctly parses and generates the code
   for variadic templates. This covers the variadic number of arguments inside
   the template brackets, new functions sizeof... and multiple inheritance
   using variadic number of classes.
*/
%module cpp11_variadic_templates
%warnfilter(SWIGWARN_CPP11_VARIADIC_TEMPLATE) MultiArgs;
%warnfilter(SWIGWARN_CPP11_VARIADIC_TEMPLATE) SizeOf;
%warnfilter(SWIGWARN_CPP11_VARIADIC_TEMPLATE) MultiInherit;

////////////////////////
// Variadic templates //
////////////////////////
%inline %{
#include <vector>
#include <string>
#include <map>

template<typename... Values>
class MultiArgs {
};

class MultiArgs<int, std::vector<int>, std::map<std::string, std::vector<int>>> multiArgs;

%}

// TODO
%template (MultiArgs1) MultiArgs<int, std::vector<int>, std::map<std::string, std::vector<int>>>;

////////////////////////
// Variadic sizeof... //
////////////////////////
%inline %{
template<typename... Args> struct SizeOf {
  static const int size = sizeof...(Args);
};
%}

%template (SizeOf1) SizeOf<int, int>;

//////////////////////////
// Variadic inheritance //
//////////////////////////
%inline %{
class A {
public:
  A() {
    a = 100;
  }
  virtual ~A() {}
  int a;
};

class B {
public:
  B() {
    b = 200;
  }
  virtual ~B() {}
  int b;
};

template <typename... BaseClasses> class MultiInherit : public BaseClasses... {
public:
  MultiInherit(BaseClasses&... baseClasses) : BaseClasses(baseClasses)... {}
  int InstanceMethod() { return 123; }
  static int StaticMethod() { return 456; }
};
%}


// TODO
//%template (MultiInherit0) MultiInherit<>;
%template (MultiInherit1) MultiInherit<A>;
// TODO
%template (MultiInherit2) MultiInherit<A,B>;

