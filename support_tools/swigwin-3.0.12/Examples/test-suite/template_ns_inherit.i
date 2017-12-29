// Submitted by Marcelo Matus
%module template_ns_inherit
 
%inline %{
  namespace hello  {
      typedef double Double;
  }
  namespace hello
  {
    template <class ArgType, class ResType>
    class VUnaryFunction
    {};
 
    template <class ArgType, class ResType>
    class UnaryFunction  : public VUnaryFunction<ArgType, ResType>
    {};
  }
 
%}
 
namespace hello
{
  %template(VUnaryFunction_id) VUnaryFunction<int, Double>;
  %template(UnaryFunction_id) UnaryFunction<int, Double>;
}                                                                             





