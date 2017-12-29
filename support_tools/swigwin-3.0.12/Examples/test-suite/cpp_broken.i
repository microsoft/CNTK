%module cpp_broken


// bug #940318
%inline %{
typedef enum {
eZero = 0
#define ONE 1
} EFoo;
%}


