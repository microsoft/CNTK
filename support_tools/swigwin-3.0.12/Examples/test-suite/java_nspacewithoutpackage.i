%module java_nspacewithoutpackage

%warnfilter(SWIGWARN_JAVA_NSPACE_WITHOUT_PACKAGE) TopLevel::Foo;
%warnfilter(SWIGWARN_JAVA_NSPACE_WITHOUT_PACKAGE) TopLevel::Bar;

%pragma(java) jniclasspackage="PragmaDefinedPackage"

SWIG_JAVABODY_PROXY(public, public, SWIGTYPE)
SWIG_JAVABODY_TYPEWRAPPER(public, public, public, SWIGTYPE)

%include <std_string.i>

%nspace TopLevel::Foo;
%nspace TopLevel::Bar;

%{
	#include <string>
%}

%inline %{

namespace TopLevel
{
  class Foo {
  public:
    virtual ~Foo() {}
    virtual std::string ping() { return "TopLevel::Foo::ping()"; }
  };

  class Bar {
  public:
    virtual ~Bar() {}
    virtual std::string pong() { return "TopLevel::Bar::pong()"; }
  };
}

%}
