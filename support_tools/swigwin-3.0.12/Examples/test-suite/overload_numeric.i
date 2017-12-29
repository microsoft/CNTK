%module overload_numeric

// Tests overloading of integral and floating point types to verify the range checking required
// for dispatch to the correct overloaded method

#ifdef SWIGLUA
// lua only has one numeric type, so most of the overloads shadow each other creating warnings
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) Nums::over;
#endif

%{
#include <iostream>
%}

%inline %{
#include <limits.h>
#include <float.h>
struct Limits {
  signed char schar_min() { return SCHAR_MIN; }
  signed char schar_max() { return SCHAR_MAX; }
  short shrt_min() { return SHRT_MIN; }
  short shrt_max() { return SHRT_MAX; }
  int int_min() { return INT_MIN; }
  int int_max() { return INT_MAX; }
  float flt_min() { return FLT_MIN; }
  float flt_max() { return FLT_MAX; }
  double dbl_max() { return DBL_MAX; }
};

struct Nums {
  const char * over(signed char v) {
    return "signed char";
  }
  const char * over(short v) {
    return "short";
  }
  const char * over(int v) {
    return "int";
  }
  const char * over(float v) {
    return "float";
  }
  const char * over(double v) {
    return "double";
  }
  double doublebounce(double v) {
    return v;
  }
};
%}

