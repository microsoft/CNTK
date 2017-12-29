// A class with a private assignment operator.
// This is rare, but sometimes used with singletons and
// objects that have complicated state.

%module private_assign
%{
#include <stdlib.h>
%}

%inline %{
   class Foo {
   private:
       Foo &operator=(const Foo &f) {
           return *this;
       }
   public:
       void bar() { }
   };

   Foo blah() {
      return Foo();
   }

  class Bar : protected Foo
  {
  };

%}

#pragma SWIG nowarn=SWIGWARN_IGNORE_OPERATOR_NEW // operator new

%inline %{
  class TROOT {
  protected:
     void *operator new(size_t l) { return malloc(sizeof(TROOT)); }
   
    int prot_meth() 
    {
      return 1;
    }
    
  public:
    TROOT()
    {
    }

    TROOT(const char *name, const char *title, void *initfunc = 0)
    {
    }
  };

  class A : protected TROOT
  {
  };
  
%}

#ifdef SWIGPYTHON

// This case only works in python
%inline %{
   struct FooBar : Foo 
   {
   };
   
   FooBar bar;
   
%}


#endif
