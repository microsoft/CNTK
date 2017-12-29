%module template_default2

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) oss::traits; // Ruby, wrong class name

%inline %{
  namespace oss
  {
    enum Polarization { UnaryPolarization, BinaryPolarization };
 
    template <Polarization P>
    struct Interface_tpl
    {
    };
 
    struct traits
    {
      static const Polarization pmode = UnaryPolarization;
    };
 
    template <class C,
          Polarization P = C::pmode,
          class Base = Interface_tpl<P> >   // **** problem here *****
    struct Module : Base
    {
    };
 
  }
%}                                

namespace oss
{
  %template(Interface_UP) Interface_tpl<UnaryPolarization>;
 
  // This works
  %template(Module_UP1) Module<traits,
                   UnaryPolarization,
                   Interface_tpl<UnaryPolarization> >;
 
   // These don't
  %template(Module_UP2) Module<traits, UnaryPolarization>;
  %template(Module_UP3) Module<traits>;
}
 
                                                       
