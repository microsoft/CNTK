%module string_simple

%newobject copy_string;

%inline %{
#include <string.h>
const char* copy_string(const char* str) {
  size_t len = strlen(str);
  char* newstring = (char*) malloc(len + 1);
  strcpy(newstring, str);
  return newstring;
}
%}
