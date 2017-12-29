#ifndef ___typedef_import_h__
#define ___typedef_import_h__

#ifdef SWIG
%module template_typedef_cplx2;
#endif

#include <complex>
typedef std::complex<double> Complex;

namespace vfncs {

  struct UnaryFunctionBase
  {
    int get_base_value()
    {
      return 0;
    }
  };    
    
  template <class ArgType, class ResType>
  struct UnaryFunction;
    
  template <class ArgType, class ResType>
  struct ArithUnaryFunction;  
    
  template <class ArgType, class ResType>
  struct UnaryFunction : UnaryFunctionBase
  {
    int get_value()
    {
      return 1;
    }
  };

  template <class ArgType, class ResType>
  struct ArithUnaryFunction : UnaryFunction<ArgType, ResType>
  {
    int get_arith_value()
    {
      return 2;
    }
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
  struct arith_traits< double, double >
  {    
    typedef double argument_type;
    typedef double result_type;
    static const char* const arg_type;
    static const char* const res_type;
  };

  template<>
  struct arith_traits< Complex, Complex >
  {
    
    typedef Complex argument_type;
    typedef Complex result_type;
    static const char* const arg_type;
    static const char* const res_type;
  };

  template<>
  struct arith_traits< Complex, double >
  {
    typedef double argument_type;
    typedef Complex result_type;
    static const char* const arg_type;
    static const char* const res_type;
  };

  template<>
  struct arith_traits< double, Complex >
  {
    typedef double argument_type;
    typedef Complex result_type;
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

const char* const arith_traits< double, double >::arg_type = "double";
const char* const arith_traits< double, double >::res_type = "double";

const char* const arith_traits< Complex, Complex >::arg_type = "complex";
const char* const arith_traits< Complex, Complex >::res_type = "complex";

const char* const arith_traits< Complex, double >::arg_type = "double";
const char* const arith_traits< Complex, double >::res_type = "complex";

const char* const arith_traits< double, Complex >::arg_type = "double";
const char* const arith_traits< double, Complex >::res_type = "complex";

#endif

} // end namespace vfncs

#ifdef SWIG

namespace vfncs {
  %template(UnaryFunction_double_double) UnaryFunction<double, double >;  
  %template(ArithUnaryFunction_double_double) ArithUnaryFunction<double, double >;  
  %template() unary_func_traits<double, double >;
  %template() arith_traits<double, double >;
  %template(make_Identity_double) make_Identity<double >;

  %template(UnaryFunction_complex_complex) UnaryFunction<Complex, Complex >;  
  %template(ArithUnaryFunction_complex_complex) ArithUnaryFunction<Complex, Complex >;  

  %template() unary_func_traits<Complex, Complex >;
  %template() arith_traits<Complex, Complex >;
  %template(make_Identity_complex) make_Identity<Complex >;

  /* [beazley] Added this part */
  %template() unary_func_traits<double,Complex>;
  %template(UnaryFunction_double_complex) UnaryFunction<double,Complex>;
  %template(ArithUnaryFunction_double_complex) ArithUnaryFunction<double,Complex>;

  /* */

  %template() arith_traits<Complex, double >;
  %template() arith_traits<double, Complex >;

  %template(make_Multiplies_double_double_complex_complex)
    make_Multiplies<double, double, Complex, Complex>;

  %template(make_Multiplies_double_double_double_double)
    make_Multiplies<double, double, double, double>;

  %template(make_Multiplies_complex_complex_complex_complex)
    make_Multiplies<Complex, Complex, Complex, Complex>;

  %template(make_Multiplies_complex_complex_double_double)
    make_Multiplies<Complex, Complex, double, double>;

}

#endif

#endif //___template_typedef_h__
