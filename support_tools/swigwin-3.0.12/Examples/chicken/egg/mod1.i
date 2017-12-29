%module mod1

%inline %{
class Bar {
  public:
    int b;
};
%}
