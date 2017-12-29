%module using_protected

%inline %{
class Foo {
protected:
  int x;
  int blah(int xx) { return xx; }
  virtual int vmethod(int xx) { return xx; }
  virtual ~Foo() {}
};

class FooBar : public Foo {
public:
  using Foo::blah;
  using Foo::x;
  using Foo::vmethod;
};

class FooBaz : public Foo {
protected:
  using Foo::blah;
  using Foo::x;
  using Foo::vmethod;
};

%}

