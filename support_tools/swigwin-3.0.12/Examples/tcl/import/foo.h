#include "base.h"

class Foo : public Base {
 public:
  Foo() { }
  ~Foo() { }
  virtual void A() { 
    printf("I'm Foo::A\n");
  }
  void B() {
    printf("I'm Foo::B\n");
  }
  virtual Base *toBase() {
    return static_cast<Base *>(this);
  }
  static Foo *fromBase(Base *b) {
    return dynamic_cast<Foo *>(b);
  }
};


