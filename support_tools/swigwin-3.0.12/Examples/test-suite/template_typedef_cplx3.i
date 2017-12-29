%module template_typedef_cplx3
%{
#include "template_typedef_cplx2.h"
%}

%include "template_typedef_cplx2.h"

%inline %{

  typedef vfncs::ArithUnaryFunction<double, double> RFunction;
  typedef vfncs::ArithUnaryFunction<Complex, Complex> CFunction;
  

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

  



