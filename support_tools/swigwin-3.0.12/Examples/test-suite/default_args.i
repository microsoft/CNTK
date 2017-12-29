// Lots of tests for methods with default parameters / default arguments

%module default_args

%{
#if defined(_MSC_VER)
  #pragma warning(disable: 4290) // C++ exception specification ignored except to indicate a function is not __declspec(nothrow)
#endif
%}

%include <std_string.i>

%inline %{
  #include <string>

  // All kinds of numbers: hex, octal (which pose special problems to Python), negative...
  void trickyvalue1(int first, int pos = -1) {}
  void trickyvalue2(int first, unsigned rgb = 0xabcdef) {}
  void trickyvalue3(int first, int mode = 0644) {}

  void doublevalue1(int first, double num = 0.0e-1) {}
  void doublevalue2(int first, double num = -0.0E2) {}

  // Long long arguments are not handled at Python level currently but still work.
  void seek(long long offset = 0LL) {}
  void seek2(unsigned long long offset = 0ULL) {}
  void seek3(long offset = 0L) {}
  void seek4(unsigned long offset = 0UL) {}
  void seek5(unsigned long offset = 0U) {}
  void seek6(unsigned long offset = 02U) {}
  void seek7(unsigned long offset = 00U) {}
  void seek8(unsigned long offset = 1U) {}
  void seek9(long offset = 1L) {}
  void seekA(long long offset = 1LL) {}
  void seekB(unsigned long long offset = 1ULL) {}

  // Anonymous arguments
  int anonymous(int = 7771);
  int anonymous(int x) { return x; }

  // Bug [548272] Default arguments
  bool booltest(bool x = true) { return x; }

  // scoped enums
  enum flavor { BITTER, SWEET };
  class EnumClass {
    public:
      enum speed { FAST, SLOW };
      // Note: default values should be EnumClass::FAST and SWEET 
      bool blah(speed s = FAST, flavor f = SWEET) { return (s == FAST && f == SWEET); };
  };

  // using base class enum in a derived class
  class DerivedEnumClass : public EnumClass {
  public:
    void accelerate(speed s = SLOW) { }
  };

  // casts
  const char * casts1(const char *m = (const char *) NULL) {
    char *ret = NULL; 
    if (m) { 
      ret = new char[strlen(m)+1];
      strcpy(ret, m);
    }
    return ret;
  }
  const char * casts2(const char *m = (const char *) "Hello") {
    char *ret = NULL; 
    if (m) { 
      ret = new char[strlen(m)+1];
      strcpy(ret, m);
    }
    return ret;
  }

  // char
  char chartest1(char c = 'x') { return c; }
  char chartest2(char c = '\0') { return c; }
  char chartest3(char c = '\1') { return c; }
  char chartest4(char c = '\n') { return c; }
  char chartest5(char c = '\102') { return c; } // 'B'
  char chartest6(char c = '\x43') { return c; } // 'C'

  // namespaces
  namespace AType { 
    enum AType { NoType }; 
  } 
  void dummy(AType::AType aType = AType::NoType) {}
  namespace A { 
    namespace B { 
      int CONST_NUM = 10; 
    } 
    int afunction(int i = B::CONST_NUM) { return i; }
  } 

  // references
  int reftest1(const int &x = 42) { return x; }
  std::string reftest2(const std::string &x = "hello") { return x; }

  // enum scope
  class Tree {
    public:
      enum types {Oak, Fir, Cedar};
      void chops(enum types type) {}
      void test(int x = Oak + Fir + Cedar) {}
  };
  enum Tree::types chops(enum Tree::types type) { return type; }
 
%}

// Rename a class member
%rename(bar2) Foo::bar;
%rename(newname) Foo::oldname(int x = 1234);
%ignore Foo::Foo(int x, int y = 0, int z = 0);
%ignore Foo::meth(int x, int y = 0, int z = 0);
%rename(renamed3arg) Foo::renameme(int x, double d) const;
%rename(renamed2arg) Foo::renameme(int x) const;
%rename(renamed1arg) Foo::renameme() const;

%typemap(default) double* null_by_default "$1=0;";

%inline %{
  typedef void* MyHandle;

  // Define a class
  class Foo {
    public:
      static int bar;
      static int spam;

      Foo(){}
     
      Foo(int x, int y = 0, int z = 0){}

      void meth(int x, int y = 0, int z = 0){}
    
      // Use a renamed member as a default argument.  SWIG has to resolve
      // bar to Foo::bar and not Foo::spam.  SWIG-1.3.11 got this wrong.
      // (Different default parameter wrapping in SWIG-1.3.23 ensures SWIG doesn't have to resolve these symbols).
      void method1(int x = bar) {}

      // Use unrenamed member as default
      void method2(int x = spam) {}

      // test the method itself being renamed
      void oldname(int x = 1234) {}
      void renameme(int x = 1234, double d=123.4) const {}

      // test default values for pointer arguments
      int double_if_void_ptr_is_null(int n, void* p = NULL) { return p ? n : 2*n; }
      int double_if_handle_is_null(int n, MyHandle h = 0) { return h ? n : 2*n; }
      int double_if_dbl_ptr_is_null(int n, double* null_by_default)
        { return null_by_default ? n : 2*n; }
  };
  int Foo::bar = 1;
  int Foo::spam = 2;
%}


// tests valuewrapper
%feature("compactdefaultargs") MyClass2::set;
%inline %{
  enum MyType { Val1, Val2 }; 

  class MyClass1 
  { 
    public: 
      MyClass1(MyType myType) {}
  }; 

  class MyClass2 
  { 
    public : 
      void set(MyClass1 cl1 = Val1) {}
      // This could have been written : set(MyClass1 cl1 = MyClass1(Val1)) 
      // But it works in C++ since there is a "conversion" constructor in  MyClass1. 
      void set2(MyClass1 cl1 = Val1) {}
  };
%}


// Default parameters with exception specifications
%inline %{
void exceptionspec(int a = -1) throw (int, const char*) {
  if (a == -1)
    throw "ciao";
  else
    throw a;
}
struct Except {
  Except(bool throwException, int a = -1) throw (int) {
    if (throwException)
      throw a;
  }
  void exspec(int a = 0) throw (int, const char*) {
    ::exceptionspec(a);
  }
};
%}

// Default parameters in static class methods
#ifdef SWIGPYTHON
%rename(staticMethod) staticmethod;
#endif

%inline %{
namespace SpaceName {
  struct Statics {
    static int staticmethod(int a=10, int b=20, int c=30) { return a+b+c; }
  };
}
%}


// Tests which could never be wrapped prior to changes in default argument wrapping implemented in SWIG-1.3.23:
%inline %{
class Tricky {
  static int getDefault() { return 500; }
  enum { privatevalue = 200 };
  static const char charvalue;
public:
  int privatedefault(int val = privatevalue) { return val; }
  int protectedint(int val = intvalue) { return val; }
  double protecteddouble(double val = doublevalue) { return val; }
  int functiondefault(int val = Tricky::getDefault()) { return val; }
  char contrived(const char *c = &charvalue) { return *c; }
protected:
  static const int intvalue = 2000;
  static const double doublevalue;
};
const char Tricky::charvalue = 'X';
const double Tricky::doublevalue = 987.654;


// tests default argument which is a constructor call within namespace
// also tests default constructor (from defaulted parameter)
namespace Space {
struct Klass {
  int val;
  Klass(int val = -1) : val(val) {}
  static Klass inc(int n = 1, const Klass& k = Klass()) { return Klass(k.val + n); }
};
Klass constructorcall(const Klass& k = Klass()) { return k; }

}
%}

%{
struct ConstMethods {
  int coo(double d = 0.0) { return 10; }
  int coo(double d = 0.0) const { return 20; }
};
%}

// const methods 
// runtime test needed to check that the const method is called
struct ConstMethods {
  int coo(double d = 0.0) const;
};



// Default args with C linkage
%inline
%{
  extern "C" double cfunc1(double x,double p = 1) {
    return(x+p);
  }

  extern "C" {
    double cfunc2(double x,double p = 2) {
      return(x+p);
    }

    double cfunc3(double x,double p = 3) {
      return(x+p);
    }

    typedef struct Pointf { 
      double		x,y; 
    } Pointf;
  }
%}

// Default arguments after ignored ones.
%typemap(in, numinputs=0) int square_error { $1 = 2; };
%typemap(default, noblock=1) int def17 { $1 = 17; };

// Enabling autodoc feature has a side effect of disabling the generation of
// aliases for functions that can hide problems with default arguments at
// Python level.
%feature("autodoc","0") slightly_off_square;

%inline %{
  inline int slightly_off_square(int square_error, int def17) { return def17*def17 + square_error; }
%}

// Python C default args
%feature("python:cdefaultargs") CDA::cdefaultargs_test1;
%inline %{
struct CDA {
  int cdefaultargs_test1(int a = 1) { return a; }
  int cdefaultargs_test2(int a = 1) { return a; }
};
%}

