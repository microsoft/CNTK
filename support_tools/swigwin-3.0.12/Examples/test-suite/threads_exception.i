// Throw a lot of exceptions

// The Python runtime tests were previously failing with the -threads option on Windows due to SWIG_PYTHON_THREAD_BEGIN_ALLOW not being within the try block.

%module(threads="1") threads_exception

%{
struct A {};
%}

%inline %{
#include <string>

#if defined(_MSC_VER)
  #pragma warning(disable: 4290) // C++ exception specification ignored except to indicate a function is not __declspec(nothrow)
#endif

class Exc {
public:
  Exc(int c, const char *m) {
    code = c;
    strncpy(msg,m,255);
    msg[255] = 0;
  }
  int code;
  char msg[256];
};

class Test {
public:
  int simple() throw(int) {
      throw(37);
      return 1;
  }
  int message() throw(const char *) {
      throw("I died.");
      return 1;
  }
  int hosed() throw(Exc) {
      throw(Exc(42,"Hosed"));
      return 1;
  } 
  int unknown() throw(A*) {
      static A a;
      throw &a;
      return 1;
  }
  int multi(int x) throw(int, const char *, Exc) {
     if (x == 1) throw(37);
     if (x == 2) throw("Bleah!");
     if (x == 3) throw(Exc(42,"No-go-diggy-die"));
     return 1;
  }
};

#ifdef SWIGPYTHON_BUILTIN
bool is_python_builtin() { return true; }
#else
bool is_python_builtin() { return false; }
#endif
%}
