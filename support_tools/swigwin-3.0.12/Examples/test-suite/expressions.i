%module expressions

%inline %{
struct A
{
    A() : k( 20/(5-1) ) {}
    A(int i) : k( 20/(5-1)*i /* comment */ ) {}
    int k;
};
%}
