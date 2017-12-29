%module template_typedef_cplx5

%{
#include <complex>
%}


%inline %{

  // This typedef triggers an inifinite recursion
  // in the next test1() nd test2() function declarations

  typedef std::complex<double> complex;  

  struct A 
  {
    complex test1() { complex r; return r; }
    std::complex<double> test2() { std::complex<double> r; return r; }
  };
  
%}

