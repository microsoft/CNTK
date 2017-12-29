%module nested_workaround
// "flatnested" emulates deprecated feature "nested_workaround" for the languages not supporting nested classes
%feature ("flatnested");

%inline %{
class Outer {
public:
  class Inner {
      int val;
    public:
      Inner(int v = 0) : val(v) {}
      void setValue(int v) { val = v; }
      int getValue() const { return val; }
  };
  Inner createInner(int v) const { return Inner(v); }
  int getInnerValue(const Inner& i) const { return i.getValue(); }
  Inner doubleInnerValue(Inner inner) { 
    inner.setValue(inner.getValue() * 2); 
    return inner;
  }
};
%}
