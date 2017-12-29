%module extend

%extend Base {
  ~Base() {
    delete $self; 
  }
  static int zeroVal() {
    return 0;
  }
  virtual int currentValue() {
    return $self->value;
  }
  int extendmethod(int v) {
    int ret = $self->method(v);
    return ret * 2;
  }
};

%inline %{
struct Base {
  Base(int v = 0) : value(v) {}
  int value;
  virtual int method(int v) {
    return v;
  }
#if !defined(SWIG)
  virtual ~Base() {}
#endif
};
struct Derived : Base {
  double actualval;
};
%}

%{
  double extendval = 0;
  double Derived_extendval_get(Derived *self) {
    return self->actualval * 100;
  }
  void Derived_extendval_set(Derived *self, double d) {
    self->actualval = d/100;
  }
%}

%extend Derived {
  Derived(int v) {
    Derived *$self = new Derived();
    $self->value = v*2;
    return $self;
  }
  virtual int method(int v) {
    int ret = $self->Base::method(v);
    return ret * 2;
  }
  double extendval;
}
