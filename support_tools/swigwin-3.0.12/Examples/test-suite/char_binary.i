/*
A test case for testing non null terminated char pointers.
*/

%module char_binary

%apply (char *STRING, size_t LENGTH) { (const char *str, size_t len) }
%apply (char *STRING, size_t LENGTH) { (const unsigned char *str, size_t len) }

%inline %{
struct Test {
  size_t strlen(const char *str, size_t len) {
    return len;
  }
  size_t ustrlen(const unsigned char *str, size_t len) {
    return len;
  }
};

typedef char namet[5];
namet var_namet;

typedef char* pchar;
pchar var_pchar;
%}

// Remove string handling typemaps and treat as pointer
%typemap(freearg) SWIGTYPE * ""
%apply SWIGTYPE * { char * }

%include "carrays.i"
%array_functions(char, pchar);

