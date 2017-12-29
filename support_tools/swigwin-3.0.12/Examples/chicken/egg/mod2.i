%module mod2

%import "mod1.i"

%{
class Bar {
  public:
    int b;
};
%}

%inline %{
  class Bar2 : public Bar {
    public:
      int c;
  };
%}
