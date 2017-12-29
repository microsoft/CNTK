%module typedef_sizet

typedef unsigned long long size_t;
%inline %{
size_t size(size_t x) {return x; } 
%}
