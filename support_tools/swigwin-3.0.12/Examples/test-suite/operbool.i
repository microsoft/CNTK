%module operbool

%rename(operator_bool) operator bool();

%inline %{
  class Test {
  public:
    operator bool() {
      return false;
    }
  };
%}
