%module xxx

template<typename T> struct Temply {
  T thing;
};

struct A {
  int var;
%template(TemplyInt) Temply<int>;
};


struct B {
  int var;
};

%extend B {
%template(TemplyDouble) Temply<double>;
}

