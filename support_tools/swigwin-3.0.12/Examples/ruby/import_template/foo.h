#include "base.h"

template<class T> class Foo : public Base<T> {
 public:
  Foo() { }
  ~Foo() { }
  virtual const char * A() const { 
    return "Foo::A";
  }
  const char * B() const {
    return "Foo::B";
  }
  virtual Base<T> *toBase() {
    return static_cast<Base<T> *>(this);
  }
  static Foo<T> *fromBase(Base<T> *b) {
    return dynamic_cast<Foo<T> *>(b);
  }
};


