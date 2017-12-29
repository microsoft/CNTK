%module extend_placement

// Tests placement of %extend directives

// Before the class

%extend Foo {
  Foo(int a) { return new Foo(); }
  ~Foo() { delete $self;}
  int spam(int x) { return x; }
  int spam(int x, int y) { return x + y ; }
  int spam(int x, int y,int z) { return x + y ; }
  int spam(Foo f, double d = 10.0) { return 0; }
};

%inline %{
class Foo {
public:
  Foo(){}

#ifdef SWIG
%extend { Foo(int a, int b) { return new Foo(); } }  
#endif

  int spam() { return 1; }
  int spam(const char* c) { return 2; }
};
%}

// After the class

%inline %{
class Bar {
public:
  Bar() { }
  int spam() { return 1; }
  int spam(const char* c) { return 2; }
};
%}


%extend Bar {
  Bar(int a) { return new Bar(); }
  ~Bar() { delete $self;}
  int spam() { return 1}
  int spam(int x) { return x; }
  int spam(int x, int y) { return x + y ; }
  int spam(int x, int y,int z) { return x + y ; }
  int spam(Bar b, double d = 10.0) { return 0; }
};


// testing templates

// Before the class

%extend FooT {
  FooT(int a) { return new FooT<T>(); }
  ~FooT() { delete $self;}
  int spam(int x) { return x; }
  int spam(int x, int y) { return x + y ; }
  int spam(int x, int y,int z) { return x + y ; }
  int spam(Foo f, double d = 10.0) { return 0; }
};

%inline %{
template<class T>
class FooT {
public:
  FooT(){}

#ifdef SWIG
%extend { FooT(int a, int b) { return new FooT<T>(); } }  
#endif

  int spam() { return 1; }
  int spam(const char* c) { return 2; }
};
%}

%template(FooTi) FooT<int>;


// After the class

%inline %{
template<class T>
class BarT {
public:
  BarT() { }
  int spam() { return 1; }
  int spam(const char* c) { return 2; }
};
%}


%extend BarT {
  BarT(int a) { return new BarT<T>(); }
  ~BarT() { delete $self;}
  int spam() { return 1}
  int spam(int x) { return x; }
  int spam(int x, int y) { return x + y ; }
  int spam(int x, int y,int z) { return x + y ; }
  int spam(Bar b, double d = 10.0) { return 0; }
};

%template(BarTi) BarT<int>;
