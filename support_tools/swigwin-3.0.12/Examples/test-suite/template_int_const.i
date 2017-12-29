%module template_int_const

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) interface_traits;	/* Ruby, wrong class name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) module_traits;		/* Ruby, wrong class name */

%inline %{ 
  enum Polarization { UnaryPolarization, BinaryPolarization }; 
  struct interface_traits 
  { 
    static const Polarization polarization = UnaryPolarization; 
  }; 
  template <Polarization P> 
    struct Interface_
    { 
    }; 
 
  typedef unsigned int Category; 
  struct module_traits 
  { 
    static const Category category = 1; 
  }; 
  
  template <Category C> 
    struct Module 
    { 
    }; 
%} 
 
%template(Interface_UP) Interface_<UnaryPolarization>; 
%template(Module_1) Module<1>; 
 
%inline %{ 
  struct ExtInterface1 :  
    Interface_<UnaryPolarization> // works 
  { 
  }; 
  struct ExtInterface2 : 
    Interface_<interface_traits::polarization>  // doesn't work 
  { 
  }; 
  struct ExtModule1 : 
    Module<1>         // works 
  { 
  }; 
  struct ExtModule2 : 
    Module<module_traits::category>    // doesn't work 
  { 
  }; 
%} 
 
