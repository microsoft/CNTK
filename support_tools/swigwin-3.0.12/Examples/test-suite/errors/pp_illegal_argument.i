%module xxx


%define MISSING_DOT1(a,
b,
..)
xxx
%enddef

%define MISSING_DOT2(..)
xxx
%enddef

%define BAD_ARGNAME(
a,
b{c
)
xxx
%enddef

