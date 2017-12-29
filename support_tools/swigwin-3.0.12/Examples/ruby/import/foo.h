#include "base.h"

class Foo : public Base {
 public:
  Foo() { }
  ~Foo() { }
  virtual const char * A() const { 
    return "Foo::A";
  }
  const char * B() const {
    return "Foo::B";
  }
  virtual Base *toBase() {
    return static_cast<Base *>(this);
  }
  static Foo *fromBase(Base *b) {
    return dynamic_cast<Foo *>(b);
  }
};


