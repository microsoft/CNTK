#include "base.h"

template<class T> class Bar : public Base<T> {
 public:
  Bar() { }
  ~Bar() { }
  virtual void A() { 
    printf("I'm Bar::A\n");
  }
  void B() {
    printf("I'm Bar::B\n");
  }
  virtual Base<T> *toBase() {
    return static_cast<Base<T> *>(this);
  }
  static Bar<T> *fromBase(Base<T> *b) {
    return dynamic_cast<Bar<T> *>(b);
  }

};


