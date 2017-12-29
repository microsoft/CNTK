%module overload_rename


%{

class Foo {
public:
  Foo(float a, float b=1.0)
  {
  }
  
  Foo(float a, int c, float b=1.0)
  {
  }
  
};

%}

%rename(Foo_int) Foo::Foo(float a, int c, float b=1.0);

class Foo {
public:
  Foo(float a, float b=1.0);
  Foo(float a, int c, float b=1.0);
};


