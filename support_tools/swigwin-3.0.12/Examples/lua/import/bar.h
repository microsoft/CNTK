#include "base.h"

class Bar : public Base {
 public:
  Bar() { }
  ~Bar() { }
  virtual const char * A() const { 
    return "Bar::A";
  }
  const char * B() const {
    return "Bar::B";
  }
  virtual Base *toBase() {
    return static_cast<Base *>(this);
  }
  static Bar *fromBase(Base *b) {
    return dynamic_cast<Bar *>(b);
  }
};


