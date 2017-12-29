%module global_scope_types

// no constructor/destructor wrappers as they do not use global scope operator which we are trying to test here
%nodefaultctor Dingaling;
%nodefaultdtor Dingaling;

%inline %{
struct Dingaling {};
typedef Dingaling DINGALING;
template <typename T> struct MyTemplate {
  T tt(T t) { return t; }
  T& ttr(T& t) { return t; }
};

#ifndef SWIG
// This is added so that the code will not compile, if the global scope operator on Dingaling is omitted in the generated code
namespace Spac {
  class Dingaling {
    Dingaling();
    Dingaling(const Dingaling& t);
    Dingaling& operator=(const Dingaling t);
  };
}
using namespace Spac;
#endif

namespace Spac {

  struct Ting {};
  typedef Ting TING;

  class Test {
  public:
  void something(::Dingaling t, ::Dingaling* pt, ::Dingaling& rt, const ::Dingaling& crt) {}
  void tsomething(MyTemplate< ::Dingaling > t1, MyTemplate< const ::Dingaling* > t2) {}
//  void usomething(::MyTemplate< ::DINGALING > t3, ::MyTemplate< ::DINGALING *> t4) {} // needs fixing
  void nothing(::Spac::Ting*, ::Spac::TING&) {}
  };

}

extern "C" void funcptrtest( void (*)(::Dingaling) ) {}
%}

