%module catches

%{
#if defined(_MSC_VER)
  #pragma warning(disable: 4290) // C++ exception specification ignored except to indicate a function is not __declspec(nothrow)
#endif
%}

%include <exception.i> // for throws(...) typemap

%catches(int, const char *, const ThreeException&) test_catches(int i);
%catches(int, ...) test_exception_specification(int i); // override the exception specification
%catches(...) test_catches_all(int i);

%inline %{
struct ThreeException {};
void test_catches(int i) {
  if (i == 1) {
    throw int(1);
  } else if (i == 2) {
    throw (const char *)"two";
  } else if (i == 3) {
    throw ThreeException();
  }
}
void test_exception_specification(int i) throw(int, const char *, const ThreeException&) {
  test_catches(i);
}
void test_catches_all(int i) {
  test_catches(i);
}
%}

