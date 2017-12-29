%module template_arg_typename

%inline %{


  template <class ArgType, class ResType>
  struct UnaryFunction 
  {
    typedef void* vptr_type;
  };

  template <class ArgType>
  struct BoolUnaryFunction : UnaryFunction<ArgType, bool>
			     
  {
    typedef UnaryFunction<ArgType, bool> base;
    BoolUnaryFunction(const typename base::vptr_type* vptrf) {}

  };


%}


%template(UnaryFunction_bool_bool) UnaryFunction<bool, bool>;
%template(BoolUnaryFunction_bool) BoolUnaryFunction<bool>;
