// Checks if calls to a method being defined in the base class, not
// overridden in the subclass, but again overridden in a class derived from
// the first subclass are dispatched correctly.
%module(directors="1") director_alternating;

%feature("director") Foo;

%inline %{
struct Foo {
  virtual ~Foo() {}
  virtual int id() {
    return 0;
  }
};

struct Bar : Foo {};

struct Baz : Bar {
  virtual int id() {
    return 2;
  }
};

// Note that even though the return value is of type Bar*, it really points to
// an instance of Baz (in which id() has been overridden).
Bar *getBar() {
  static Baz baz;
  return &baz;
}

// idFromGetBar() obviously is equivalent to getBar()->id() in C++ – this
// should be true from the target language as well.
int idFromGetBar() {
  return getBar()->id();
}
%}
