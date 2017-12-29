#include "bar.h"

class Spam : public Bar {
 public:
  Spam() { }
  ~Spam() { }
  virtual void A() { 
    printf("I'm Spam::A\n");
  }
  void B() {
    printf("I'm Spam::B\n");
  }
  virtual Base *toBase() {
    return static_cast<Base *>(this);
  }
  virtual Bar *toBar() {
    return static_cast<Bar *>(this);
  }
  static Spam *fromBase(Base *b) {
    return dynamic_cast<Spam *>(b);
  }
};


