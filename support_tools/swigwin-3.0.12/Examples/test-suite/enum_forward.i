%module enum_forward

/* This contains C code that is not valid C++03 and Octave, and Javascript(v8) wrappers are always compiled as C++ */
#if !defined(SWIGOCTAVE) && !defined(SWIG_JAVASCRIPT_V8)
%{
enum ForwardEnum1 { AAA, BBB };
enum ForwardEnum2 { CCC, DDD };
%}

%inline %{
#if __GNUC__ >= 5 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8)
/* ISO C forbids forward references to ‘enum’ types [-Werror=pedantic] */
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

#if !defined(__SUNPRO_C)
enum ForwardEnum1;
enum ForwardEnum2;
enum ForwardEnum2;
enum ForwardEnum3;
#endif
%}

%inline %{
enum ForwardEnum1 get_enum1() { return AAA; }
enum ForwardEnum1 test_function1(enum ForwardEnum1 e) {
  return e;
}
%}

%inline %{
enum ForwardEnum2 get_enum2() { return CCC; }
enum ForwardEnum2 test_function2(enum ForwardEnum2 e) {
  return e;
}
%}

%inline %{
enum ForwardEnum3 { EEE, FFF };
enum ForwardEnum3 get_enum3() { return EEE; }
enum ForwardEnum3 test_function3(enum ForwardEnum3 e) {
  return e;
}
%}

%inline %{
#if !defined(__SUNPRO_C)
enum ForwardEnum2;
enum ForwardEnum3;
#endif
%}

#endif
