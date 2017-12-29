%module symbol_clash

// ::Vector and ::Text::Vector were incorrectly clashing in the target language symbol tables

#if defined(SWIGJAVA) || defined(SWIGCSHARP)

#if defined(SWIGJAVA)
%include "enumtypeunsafe.swg"
#elif defined(SWIGCSHARP)
%include "enumsimple.swg"
#endif

%inline %{
class Vector
{
};

namespace Text
{
  enum Preference
  {
    Raster,
    Vector
  };
}
%}

#endif
