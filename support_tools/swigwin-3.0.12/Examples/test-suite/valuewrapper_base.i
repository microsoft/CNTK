%module valuewrapper_base
%inline 
%{ 
  namespace oss 
  { 
    enum Polarization { UnaryPolarization, BinaryPolarization }; 
 
    struct Base 
    { 
    };    
 
    template <Polarization P> 
    struct Interface_ : Base 
    { 
      Interface_(const Base& b) { }; 
    }; 
    
    template <class Result> 
    Result make() { return Result(*new Base()); }
  } 
%} 
 
namespace oss 
{ 
  // Interface 
  %template(Interface_BP) Interface_<BinaryPolarization>; 
  %template(make_Interface_BP) make<Interface_<BinaryPolarization> >; 
} 
