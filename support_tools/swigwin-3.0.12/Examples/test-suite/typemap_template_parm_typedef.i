%module typemap_template_parm_typedef

%typemap(in) SWIGTYPE " _in_will_not_compile_ "
%typemap(in) SWIGTYPE * " _in_will_not_compile_ "

%typemap(out) SWIGTYPE " _out_will_not_compile_ "
%typemap(out) SWIGTYPE * " _out_will_not_compile_ "

%{
#include <vector>
#include <list>
#include <deque>

  namespace jada {
    typedef unsigned int uint;
    void test_no_typedef(std::list<unsigned int> bada) {}
    void test_typedef(std::vector<uint> bada) {}
    std::deque<unsigned int> no_typedef_out() {
      std::deque<unsigned int> x;
      return x;
    }
  }
%}

%typemap(in) std::list<unsigned int> (std::list<unsigned int> tmp) {
  $1 = tmp;
}

%typemap(in) std::vector<unsigned int> (std::vector<unsigned int> tmp) {
  $1 = tmp;
}

%typemap(out) std::list<unsigned int> {
}

// The presennce of this 'out' typemap was hiding the std::vector<unsigned int> 'in' typemap in swig-2.0.5 and swig-2.0.6
%typemap(out) std::vector<jada::uint> {
}

// This typemap was not used for no_typedef_out in 2.0.4 and earlier
#if defined(SWIGJAVA) || defined(SWIGCSHARP)
%typemap(out) std::deque<jada::uint> {
  $result = 0;
}
#else
%typemap(out) std::deque<jada::uint> {
}
#endif

namespace jada {
  typedef unsigned int uint;
  void test_no_typedef(std::list<uint> bada);
  void test_typedef(std::vector<uint> bada);
  std::deque<unsigned int> no_typedef_out();
}

