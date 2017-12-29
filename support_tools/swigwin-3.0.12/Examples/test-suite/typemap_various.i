%module typemap_various

// %copyctor need to be disables since 'const SWIGTYPE &' is intended to generate errors
%nocopyctor;

%typemap(in) SWIGTYPE "_this_will_not_compile_SWIGTYPE_"
%typemap(in) const SWIGTYPE & "_this_will_not_compile_const_SWIGTYPE_REF_"

%inline %{
template <class T> struct Foo {
  Foo() {}
#ifdef SWIG
  // These typemaps should be used by foo1 and foo2
  %typemap(in) Foo<T>      "/*in typemap for Foo<T> */"
  %typemap(in) const Foo & "/*in typemap for const Foo&, with type T*/"
#endif
};
%}

%template(FooInt) Foo<int>;
%template() Foo<short>; // previously Foo<short> typemaps were being picked up for Python only

%inline %{
void foo1(Foo<int> f, const Foo<int>& ff) {}
void foo2(Foo<short> f, const Foo<short>& ff) {}
%}

// Check "ret" typemap is implemented
%{
  template<typename T> struct NeededForTest {};
%}
%fragment("NeededForTest", "header") %{
  template<> struct NeededForTest<short> { NeededForTest(short) {} };
%}

%typemap(ret) short "_ret_typemap_for_short_no_compile"
%typemap(ret, fragment="NeededForTest") short Bar1::foofunction() { /* ret typemap for short */ NeededForTest<short> needed($1); };
%typemap(ret, fragment="NeededForTest") short globalfoofunction() { /* ret typemap for short */ NeededForTest<short> needed($1); };

%inline %{
  struct Bar1 {
    short foofunction() { return 1;}
  };
  short globalfoofunction() { return 1;}
%}

%{
void CheckRetTypemapUsed() {
  // If the "ret" typemap is not used, the NeededForTest template specialization will not have been
  // generated and so the following code will result in a compile failure
  NeededForTest<short> needed(111);
  (void)needed;
}
%}

%newobject FFoo::Bar(bool) const ;
%typemap(newfree) char* Bar(bool)  {
   /* hello */ delete[] result;
}

%inline {
  class FFoo {
  public:
    char * Bar(bool b) const { return (char *)"x"; }
  };
}

#ifdef SWIGCSHARP
#define TYPEMAP_OUT_INIT $result = 0;
#else
#define TYPEMAP_OUT_INIT
#endif

// Test obscure bug where named typemaps where not being applied when symbol name contained a number
%typemap(out) double "_out_typemap_for_double_no_compile_"
%typemap(out) double ABCD::meth {$1 = 0.0; TYPEMAP_OUT_INIT}
%typemap(out) double ABCD::m1   {$1 = 0.0; TYPEMAP_OUT_INIT}
%typemap(out) double ABCD::_x2  {$1 = 0.0; TYPEMAP_OUT_INIT}
%typemap(out) double ABCD::y_   {$1 = 0.0; TYPEMAP_OUT_INIT}
%typemap(out) double ABCD::_3   {$1 = 0.0; TYPEMAP_OUT_INIT}
%inline %{
struct ABCD {
  double meth() { return 0.0; }
  double m1() { return 0.0; }
  double _x2() { return 0.0; }
  double y_() { return 0.0; }
  double _3() { return 0.0; }
};
%}
