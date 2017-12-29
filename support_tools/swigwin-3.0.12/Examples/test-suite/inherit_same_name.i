%module inherit_same_name

%inline %{
  struct Base {
    Base() : MethodOrVariable(0) {}
    virtual ~Base() {}
  protected:
    int MethodOrVariable;
  };
  struct Derived : Base {
    virtual void MethodOrVariable() { Base::MethodOrVariable = 10; }
  };
  struct Bottom : Derived {
    void MethodOrVariable() {}
  };
%}
