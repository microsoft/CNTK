%module template_typedef_cplx4
%{
#include "template_typedef_cplx2.h"
%}

%include "template_typedef_cplx2.h"

%inline %{

  typedef vfncs::ArithUnaryFunction<double, double> RFunction;
  // **** these two work ****
  // typedef vfncs::ArithUnaryFunction<Complex, Complex > CFunction;
  // typedef vfncs::ArithUnaryFunction<std::complex<double>, std::complex<double> > CFunction;
  
  // **** these ones don't ***
  // typedef vfncs::ArithUnaryFunction<Complex, std::complex<double> > CFunction;
  typedef vfncs::ArithUnaryFunction<std::complex<double>, Complex > CFunction;



  int my_func_r(RFunction* hello)
    {
      return 0;
    }
  
  int my_func_c(CFunction* hello)
    {
      return 1;
    }  

  struct Sin : RFunction
  {
  };  

  struct CSin : CFunction
  {
  };  
  
%}

  



