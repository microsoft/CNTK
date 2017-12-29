#include "base.h"

template<class T> class Bar : public Base<T> {
 public:
  Bar() { }
  ~Bar() { }
  virtual const char * A() const { 
    return "Bar::A";
  }
  const char * B() const {
    return "Bar::B";
  }
  virtual Base<T> *toBase() {
    return static_cast<Base<T> *>(this);
  }
  static Bar<T> *fromBase(Base<T> *b) {
    return dynamic_cast<Bar<T> *>(b);
  }

};


