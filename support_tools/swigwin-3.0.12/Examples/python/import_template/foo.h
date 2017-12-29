#include "base.h"

template<class T> class Foo : public Base<T> {
 public:
  Foo() { }
  ~Foo() { }
  virtual void A() { 
    printf("I'm Foo::A\n");
  }
  void B() {
    printf("I'm Foo::B\n");
  }
  virtual Base<T> *toBase() {
    return static_cast<Base<T> *>(this);
  }
  static Foo<T> *fromBase(Base<T> *b) {
    return dynamic_cast<Foo<T> *>(b);
  }
};


