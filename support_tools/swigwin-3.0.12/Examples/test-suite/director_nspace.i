%module(directors="1") director_nspace

#ifdef SWIGJAVA
SWIG_JAVABODY_PROXY(public, public, SWIGTYPE)
SWIG_JAVABODY_TYPEWRAPPER(public, public, public, SWIGTYPE)
#endif

%{
#include <string>

namespace TopLevel
{
  namespace Bar
  {
    class FooBar {
    public:
      FooBar() {}
      FooBar(const FooBar&) {}
      virtual ~FooBar() {}
      
      std::string FooBarDo() { return "Bar::Foo2::Foo2Bar()"; }
    };
    
    class Foo {
    public:
      virtual ~Foo() {}
      virtual std::string ping() { return "Bar::Foo::ping()"; }
      virtual std::string pong() { return "Bar::Foo::pong();" + ping(); }
      virtual std::string fooBar(FooBar* fb) { return fb->FooBarDo(); }
      virtual Foo makeFoo() { return Foo(); }
      virtual FooBar makeFooBar() { return FooBar(); }
  
      static Foo* get_self(Foo *self_) {return self_;}
    };
  }
}

%}

%include <std_string.i>

// nspace feature only supported by these languages
#if defined(SWIGJAVA) || defined(SWIGCSHARP) || defined(SWIGD) || defined(SWIGLUA) || defined(SWIGJAVASCRIPT)
%nspace TopLevel::Bar::Foo;
%nspace TopLevel::Bar::FooBar;
#else
//#warning nspace feature not yet supported in this target language
#endif

%feature("director") TopLevel::Bar::Foo;

namespace TopLevel
{
  namespace Bar
  { 
    class FooBar {
    public:
      FooBar();
      FooBar(const FooBar&);
      virtual ~FooBar();
      
      std::string FooBarDo();
      
    };
    
    class Foo
    {
    public:
      virtual ~Foo();
      virtual std::string ping();
      virtual std::string pong();
      virtual std::string fooBar(FooBar* fb);
      virtual Foo makeFoo();
      virtual FooBar makeFooBar();
     
      static Foo* get_self(Foo *self_);
    };
  }
}
