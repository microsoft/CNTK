%module smart_pointer_member

%warnfilter(SWIGWARN_GO_NAME_CONFLICT);                       /* Ignoring 'foo' due to Go name ('Foo') conflict with 'Foo' */

%inline %{

  class Foo {
  public:
    int x[4];
    int y;
    static const int z;
    static const int ZZ = 3;
    static int zx;

    static int boo() { return 0;}

    friend int foo(Foo* foo) { return 0;}
  };
  
  class Bar {
    Foo *f;
  public:
    Bar(Foo *f) : f(f) { }
    Foo *operator->() {
      return f;
    }

    static int bua() { return 0;}
  };

  class CBar {
    Foo *f;
  public:
    CBar(Foo *f) : f(f) { }
    const Foo *operator->()  {
      return f;
    }
  };

  
  int get_y(Bar *b) 
  {
    return (*b)->y;
  }
  
  int get_z(Bar *b) 
  {
    return (*b)->z;
  }
%}


%{
  const int Foo::z = 3;
  int Foo::zx;
%}
