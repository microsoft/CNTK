#include "bar.h"

template<class T> class Spam : public Bar<T> {
 public:
  Spam() { }
  ~Spam() { }
  virtual const char * A() const { 
    return "Spam::A";
  }
  const char * B() const {
    return "Spam::B";
  }
  virtual Base<T> *toBase() {
    return static_cast<Base<T> *>(this);
  }
  virtual Bar<T> *toBar() {
    return static_cast<Bar<T> *>(this);
  }
  static Spam<T> *fromBase(Base<T> *b) {
    return dynamic_cast<Spam<T> *>(b);
  }
};


