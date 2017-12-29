%module template_enum_ns_inherit
%inline %{
 
  namespace oss
  {
    enum Polarization { UnaryPolarization, BinaryPolarization };
 
    template <Polarization P>
    struct Interface_
    {
    };

    template <Polarization P, class C>
    struct Module
    {
    };

  }
 
%}                                                 
 
namespace oss
{
  %template(Interface_UP) Interface_<UnaryPolarization>;
  %template(Module_UPIUP) Module<UnaryPolarization,Interface_<UnaryPolarization> >;
}
 
%inline %{
  namespace oss
  {
    namespace hello
    {
      struct HInterface1 :
           Interface_<oss::UnaryPolarization>  // this works (with fullns qualification)
      {
      };
 
      struct HInterface2 :
          Interface_<UnaryPolarization>       // this doesn't work
      {
      };
 
     struct HModule1 : Module<UnaryPolarization, Interface_<UnaryPolarization> > {
 };

    }
  }
%}                                   
