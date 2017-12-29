%module xxx

%define foo(a,x)
int ii;
%enddef

%inline %{
struct Struct {
foo(2,
};
%}
