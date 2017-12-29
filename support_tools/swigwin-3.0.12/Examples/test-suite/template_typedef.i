#ifdef SWIGPYTHON
%module("templatereduce") template_typedef
#else
%module template_typedef
#endif
//
// Change this to #if 1 to test the 'test'
//
#if 0

#define reald double
%{
#define reald double
%}

#else

%inline %{
  typedef double reald;
%}

#endif


%inline %{

  //  typedef double reald;

  namespace vfncs {

    struct UnaryFunctionBase
    {
    };    
    
    template <class ArgType, class ResType>
    struct UnaryFunction;
    
    template <class ArgType, class ResType>
    struct ArithUnaryFunction;  
    
    template <class ArgType, class ResType>
    struct UnaryFunction : UnaryFunctionBase
    {
    };

    template <class ArgType, class ResType>
    struct ArithUnaryFunction : UnaryFunction<ArgType, ResType>
    {
    };      
    
    template <class ArgType, class ResType>         
    struct unary_func_traits 
    {
      typedef ArithUnaryFunction<ArgType, ResType > base;
    };
  
    template <class ArgType>
    inline
    typename unary_func_traits< ArgType, ArgType >::base
    make_Identity()
    {
      return typename unary_func_traits< ArgType, ArgType >::base();
    }

    template <class Arg1, class Arg2>
    struct arith_traits 
    {
    };

    template<>
    struct arith_traits< float, float >
    {
    
      typedef float argument_type;
      typedef float result_type;
      static const char* const arg_type;
      static const char* const res_type;
    };

    template<>
    struct arith_traits< reald, reald >
    {
    
      typedef reald argument_type;
      typedef reald result_type;
      static const char* const arg_type;
      static const char* const res_type;
    };

    template<>
    struct arith_traits< reald, float >
    {
      typedef float argument_type;
      typedef reald result_type;
      static const char* const arg_type;
      static const char* const res_type;
    };

    template<>
    struct arith_traits< float, reald >
    {
      typedef float argument_type;
      typedef reald result_type;
      static const char* const arg_type;
      static const char* const res_type;
    };

    template <class AF, class RF, class AG, class RG>
    inline
    ArithUnaryFunction<typename arith_traits< AF, AG >::argument_type,
		       typename arith_traits< RF, RG >::result_type >
    make_Multiplies(const ArithUnaryFunction<AF, RF>& f,
		    const ArithUnaryFunction<AG, RG >& g)
    {
      return 
	ArithUnaryFunction<typename arith_traits< AF, AG >::argument_type,
	                   typename arith_traits< RF, RG >::result_type>();
    }

#ifndef SWIG

    // Initialize these static class members

    const char* const arith_traits< float, float >::arg_type = "float";
    const char* const arith_traits< float, float >::res_type = "float";

    const char* const arith_traits< reald, reald >::arg_type = "reald";
    const char* const arith_traits< reald, reald >::res_type = "reald";

    const char* const arith_traits< reald, float >::arg_type = "float";
    const char* const arith_traits< reald, float >::res_type = "reald";

    const char* const arith_traits< float, reald >::arg_type = "float";
    const char* const arith_traits< float, reald >::res_type = "reald";

#endif

  }
%}

namespace vfncs {  
  %template(UnaryFunction_float_float) UnaryFunction<float, float >;  
  %template(ArithUnaryFunction_float_float) ArithUnaryFunction<float, float >;  
  %template() unary_func_traits<float, float >;
  %template() arith_traits<float, float >;
  %template(make_Identity_float) make_Identity<float >;

  %template(UnaryFunction_reald_reald) UnaryFunction<reald, reald >;  
  %template(ArithUnaryFunction_reald_reald) ArithUnaryFunction<reald, reald >;  

  %template() unary_func_traits<reald, reald >;
  %template() arith_traits<reald, reald >;
  %template(make_Identity_reald) make_Identity<reald >;

  /* [beazley] Added this part */
  %template() unary_func_traits<float,reald>;
  %template(UnaryFunction_float_reald) UnaryFunction<float,reald>;
  %template(ArithUnaryFunction_float_reald) ArithUnaryFunction<float,reald>;

  /* */

  %template() arith_traits<reald, float >;
  %template() arith_traits<float, reald >;
  %template() arith_traits<float, float >;

  %template(make_Multiplies_float_float_reald_reald)
    make_Multiplies<float, float, reald, reald>;

  %template(make_Multiplies_float_float_float_float)
    make_Multiplies<float, float, float, float>;

  %template(make_Multiplies_reald_reald_reald_reald)
    make_Multiplies<reald, reald, reald, reald>;

}

#ifdef SWIGPYTHON
swig_type_info *SWIG_TypeQuery(const char* name);
#endif
