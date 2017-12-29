%module(directors="1",dirprot="1") director_nested
#pragma SWIG nowarn=SWIGWARN_TYPEMAP_THREAD_UNSAFE,SWIGWARN_TYPEMAP_DIRECTOROUT_PTR

%{
#include <string>
#include <iostream>
%}

%include "std_string.i"

%feature("director") Bar;
%feature("director") Foo<int>;
%feature("director") FooBar<int>;

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
    
  protected:
    std::string do_advance() 
    {
      return "Bar::do_advance;" + do_step();
    }
    

#if defined(SWIGPYTHON) || defined(SWIGRUBY) || \
  defined(SWIGJAVA) || defined(SWIGOCAML) || defined(SWIGCSHARP)
    virtual std::string do_step() const = 0;
#else
    virtual std::string do_step() const {return "";};
#endif
  };  
  
  template <class C>
    class FooBar : public Bar
    {
    public:
      virtual C get_value() const = 0;

      virtual const char * get_name() 
      {
	return "FooBar::get_name";
      }

      const char *name() 
      {
	return get_name();
      }
      
      static FooBar *get_self(FooBar *a)
      {
	return a;
      }
      
    };
}

%template(FooBar_int) FooBar<int>;

