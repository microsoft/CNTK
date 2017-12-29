%module template_enum

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) foo<int>;    /* Ruby, wrong class name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) foo<double>; /* Ruby, wrong class name */

%inline %{
template<class T> class foo {
public:
    enum { FOO, BAR };
};
%}

%template(foo_i) foo<int>;
%template(foo_d) foo<double>;

#ifdef SWIGD
// Workaround for the D module which uses the literal value in the generated wrapper code.
%dconstvalue("3") Manta::ColorSpace<Manta::RGBTraits>::NumComponents;
#endif

%inline {
  
namespace Manta {
  template<typename Traits> class ColorSpace {
  public:
    typedef typename Traits::ComponentType ComponentType;
    enum { NumComponents = Traits::NumComponents};

    ComponentType data[NumComponents];
  };
}

namespace Manta {
  class RGBTraits {
  public:
    typedef float ComponentType;
    enum {NumComponents = 3};
  };
}
}


namespace Manta {
  %template(Color) ColorSpace<RGBTraits>;
}; 
