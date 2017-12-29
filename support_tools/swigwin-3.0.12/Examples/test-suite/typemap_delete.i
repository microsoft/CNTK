%module typemap_delete

%typemap(in) Rect* (Rect temp) {
  $1 = 0;
  will_not_compile
}

%typemap(in) Rect*;

%inline %{
struct Rect
{
  int val;
  Rect(int v) : val(v) {}
};
%}
