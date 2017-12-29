%module cast_operator

%rename(tochar) A::operator char*() const;
%inline %{
#include <string.h>
struct A 
{ 
operator char*() const; 
}; 

inline 
A::operator char*() const 
{
  static char hi[16];
  strcpy(hi, "hi");
  return hi;
} 

%}

