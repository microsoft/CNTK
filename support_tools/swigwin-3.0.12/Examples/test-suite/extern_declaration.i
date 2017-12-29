%module extern_declaration 

// Test different calling conventions on Windows. Old versions of SWIG generated
// an incorrect extern declaration that wouldn't compile with Windows compilers.
#define SWIGEXPORT
#define SWIGSTDCALL
#define MYDLLIMPORT

%{
#if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
#  define MYDLLIMPORT __declspec(dllimport)
#else
#  define MYDLLIMPORT
#endif
%}

MYDLLIMPORT extern int externimport(int i);
SWIGEXPORT extern int externexport(int);
extern int SWIGSTDCALL externstdcall(int);

%{
/*
  externimport ought to be using MYDLLIMPORT and compiled into another dll, but that is 
  a bit tricky to do in the test framework
*/
SWIGEXPORT extern int externimport(int i) { return i; }
SWIGEXPORT extern int externexport(int i) { return i; }
extern int SWIGSTDCALL externstdcall(int i) { return i; }
%}


