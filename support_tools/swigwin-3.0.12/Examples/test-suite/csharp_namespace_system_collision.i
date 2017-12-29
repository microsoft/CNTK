%module namespace_system_collision

%{
#include <string>

namespace TopLevel
{
  namespace System
  {
    class Foo {
    public:
      virtual ~Foo() {}
      virtual std::string ping() { return "TopLevel::System::Foo::ping()"; }
    };
  }
}

%}

%include <std_string.i>

// nspace feature only supported by these languages
#if defined(SWIGJAVA) || defined(SWIGCSHARP) || defined(SWIGD)
%nspace;
#else
//#warning nspace feature not yet supported in this target language
#endif

namespace TopLevel
{
  namespace System
  {
    class Foo {
    public:
      virtual ~Foo();
      virtual std::string ping();
    };
  }
}
