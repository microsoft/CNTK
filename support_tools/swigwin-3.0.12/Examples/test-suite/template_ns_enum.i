%module template_ns_enum
%inline %{
  namespace hello {
    enum Hello { Hi, Hola };
 
    template <Hello H>
    struct traits
    {
      typedef double value_type;
    };
 
    traits<Hi>::value_type say_hi()
    {
      return traits<Hi>::value_type(1);
    }
 
  }
%}                                             
