#include "bar.h"

class Spam : public Bar {
 public:
  Spam() { }
  ~Spam() { }
  virtual const char * A() const { 
    return "Spam::A";
  }
  const char * B() const {
    return "Spam::B";
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


