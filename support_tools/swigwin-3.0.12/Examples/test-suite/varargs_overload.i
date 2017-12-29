// Tests SWIG's *default* handling of overloading varargs (function varargs, not preprocessor varargs).
// The default behavior is to simply ignore the varargs.
%module varargs_overload

%inline %{
#include <stdio.h>

const char *vararg_over1(const char *fmt, ...) {
  return fmt;
}
const char *vararg_over1(int i) {
  static char buffer[256];
  sprintf(buffer, "%d", i);
  return buffer;
}

const char *vararg_over2(const char *fmt, ...) {
  return fmt;
}
const char *vararg_over2(int i, double j) {
  static char buffer[256];
  sprintf(buffer, "%d %g", i, j);
  return buffer;
}

const char *vararg_over3(const char *fmt, ...) {
  return fmt;
}
const char *vararg_over3(int i, double j, const char *s) {
  static char buffer[256];
  sprintf(buffer, "%d %g %s", i, j, s);
  return buffer;
}
%}

%varargs(int mode = 0) vararg_over4;
%inline %{
const char *vararg_over4(const char *fmt, ...) {
  return fmt;
}
const char *vararg_over4(int i) {
  static char buffer[256];
  sprintf(buffer, "%d", i);
  return buffer;
}
%}
