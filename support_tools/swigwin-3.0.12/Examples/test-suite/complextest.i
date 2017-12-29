%module complextest

%include <complex.i>

#ifdef __cplusplus
%{
#include <algorithm>
#include <functional>
#include <numeric>
%}
%include <std_vector.i>

#if 1
%template(VectorStdCplx) std::vector<std::complex<double> >;
#endif

%inline 
{
  std::complex<double> Conj(const std::complex<double>&  a) 
  {
    return std::conj(a);
  }  

  std::complex<float> Conjf(const std::complex<float>&  a) 
  {
    return std::conj(a);
  }  

#if 1
  std::vector<std::complex<double> > Copy_h(const std::vector<std::complex<double> >&  a) 
  {
    std::vector<std::complex<double> > b(a.size()/2);
    std::copy(a.begin(), a.begin()+a.size()/2, b.begin());
    return b;
  }  
#endif
}


#else


%{
%}

%inline 
{
  complex Conj(complex a)
  {
    return conj(a);
  }


  complex float Conjf(float complex a)
  {
    return conj(a);
  }
}


#endif
