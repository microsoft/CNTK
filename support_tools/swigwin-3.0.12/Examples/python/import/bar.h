#include "base.h"

class Bar : public Base {
 public:
  Bar() { }
  ~Bar() { }
  virtual void A() { 
    printf("I'm Bar::A\n");
  }
  void B() {
    printf("I'm Bar::B\n");
  }
  virtual Base *toBase() {
    return static_cast<Base *>(this);
  }
  static Bar *fromBase(Base *b) {
    return dynamic_cast<Bar *>(b);
  }

};


