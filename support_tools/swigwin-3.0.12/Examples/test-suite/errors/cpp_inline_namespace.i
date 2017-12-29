%module xxx

namespace foo {
%inline %{
int bar(int x) { }
%}
}
