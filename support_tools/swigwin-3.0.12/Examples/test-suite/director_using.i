%module(directors="1",dirprot="1") director_using

%warnfilter(SWIGWARN_PHP_PUBLIC_BASE) FooBar;

%{
#include <string>
#include <iostream>
%}

%include "std_string.i"

%feature("director");

%newobject *::create();

%inline {
  template <class C>
    class Foo {
    public:
      virtual ~Foo() {}
      
      std::string advance() 
      {
	return "Foo::advance;" + do_advance();
      }  

    protected:
      virtual std::string do_advance() = 0;
      virtual std::string do_step() const = 0;
    };
}

%template(Foo_int) Foo<int>;

%inline {

  class Bar : public Foo<int>
  {
  public:
    
    std::string step() 
    {
      return "Bar::step;" + advance();    
    }

    
    using Foo<int>::do_step;
  protected:
    std::string do_advance() 
    {
      return "Bar::do_advance;" + do_step();
    }
    
  };  
  
  template <class C>
    class FooBar : public Bar
    {
    public:
      virtual C get_value() const = 0;
      using Bar::do_advance;
    };

}

%template(FooBar_int) FooBar<int>;

%inline %{
  struct SomeBase {
    virtual ~SomeBase() {}
    virtual void method1() {}
    virtual void method2() {}
  };

  struct PrivateDerived : SomeBase {
  private:
    virtual void method1() {}
    using SomeBase::method2;
  };
%}
