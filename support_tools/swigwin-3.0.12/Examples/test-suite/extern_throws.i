%module extern_throws 

%{
#if defined(_MSC_VER)
  #pragma warning(disable: 4290) // C++ exception specification ignored except to indicate a function is not __declspec(nothrow)
#endif
%}

%inline %{
#include <exception>
extern int get() throw(std::exception);

%}

%{
int get() throw(std::exception) { return 0; }
%}

