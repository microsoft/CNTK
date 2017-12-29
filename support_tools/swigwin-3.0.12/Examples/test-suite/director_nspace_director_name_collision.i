%module(directors="1") director_nspace_director_name_collision

#ifdef SWIGJAVA
SWIG_JAVABODY_PROXY(public, public, SWIGTYPE)
SWIG_JAVABODY_TYPEWRAPPER(public, public, public, SWIGTYPE)
#endif

%{
#include <string>

namespace TopLevel
{
  namespace A
  {
    class Foo {
    public:
      virtual ~Foo() {}
      virtual std::string ping() { return "TopLevel::A::Foo::ping()"; }
    };
  }

  namespace B
  {
    class Foo {
    public:
      virtual ~Foo() {}
      virtual std::string ping() { return "TopLevel::B:Foo::ping()"; }
    };
  }
}

%}

%include <std_string.i>

// nspace feature only supported by these languages
#if defined(SWIGJAVA) || defined(SWIGCSHARP) || defined(SWIGD) || defined(SWIGLUA) || defined(SWIGJAVASCRIPT)
%nspace TopLevel::A::Foo;
%nspace TopLevel::B::Foo;
#else
//#warning nspace feature not yet supported in this target language
%ignore TopLevel::B::Foo;
#endif

%feature("director") TopLevel::A::Foo;
%feature("director") TopLevel::B::Foo;

namespace TopLevel
{
  namespace A
  {
    class Foo {
    public:
      virtual ~Foo();
      virtual std::string ping();
    };
  }

  namespace B
  {
    class Foo {
    public:
      virtual ~Foo();
      virtual std::string ping();
    };
  }
}
