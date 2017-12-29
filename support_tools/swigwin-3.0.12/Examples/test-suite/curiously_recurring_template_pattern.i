%module curiously_recurring_template_pattern

// Test Curiously Recurring Template Pattern - CRTP

%inline %{
template <typename T> class Base {
  int base1Param;
  int base2Param;
public:
  Base() : base1Param(0) {}
  int getBase1Param() {
    return base1Param;
  }
  T& setBase1Param(int value) {
    base1Param = value;
    return *static_cast<T*>(this);
  }
  int getBase2Param() {
    return base2Param;
  }
  T& setBase2Param(int value) {
    base2Param = value;
    return *static_cast<T*>(this);
  }
  virtual ~Base() {}
};
%}

%template(basederived) Base<Derived>;

%inline %{
class Derived : public Base<Derived> {
  int derived1Param;
  int derived2Param;
public:
  Derived() : derived1Param(0), derived2Param(0) {}
  int getDerived1Param() {
    return derived1Param;
  }
  Derived& setDerived1Param(int value) {
    derived1Param = value;
    return *this;
  }
  int getDerived2Param() {
    return derived2Param;
  }
  Derived& setDerived2Param(int value) {
    derived2Param = value;
    return *this;
  }
};
%}



