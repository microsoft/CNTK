%module(ruby_minherit="1") template_inherit_abstract

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) oss::test;	/* Ruby, wrong class name */

%warnfilter(SWIGWARN_JAVA_MULTIPLE_INHERITANCE,
	    SWIGWARN_CSHARP_MULTIPLE_INHERITANCE,
	    SWIGWARN_D_MULTIPLE_INHERITANCE,
	    SWIGWARN_PHP_MULTIPLE_INHERITANCE) oss::Module;	/* C#, D, Java, PHP multiple inheritance */

%inline %{ 
 
  namespace oss 
  { 
      template <class C> 
      struct Wrap 
      { 
      }; 
 
      struct ModuleBase 
      { 
          virtual ~ModuleBase() {}
          virtual int get() = 0; 
      };      
    
      template <class C> 
      struct Module : C, ModuleBase 
      { 
	virtual ~Module() {}

        protected: 
        Module() {}
      }; 
      
      template <class  C> 
      struct HModule : Module<Wrap<C> > 
      { 
       //  virtual int get();   // declaration here works 
   
      protected: 
        HModule() {}
      }; 
  } 
 
  struct B 
  { 
  }; 
  
%}                                                 
  
namespace oss 
{ 
  %template(Wrap_B) Wrap<B>; 
  %template(Module_B) Module<Wrap<B> >; 
  %template(HModule_B) HModule<B>; 
} 
  
%inline %{ 
  namespace oss 
  { 
#if defined(SWIG) && (defined(SWIGCSHARP) || defined(SWIGD))
%ignore HModule<B>::get(); // Work around for lack of multiple inheritance support - base ModuleBase is ignored.
#endif
    struct test : HModule<B> 
    { 
    virtual int get() {return 0;}   // declaration here breaks swig 
    }; 
  } 
%}  
