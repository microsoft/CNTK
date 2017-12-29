%module li_std_vectora

%include std_vectora.i


%{
#include <algorithm>
#include <functional>
#include <numeric>
%}


%template(vector_i) std::vector<int, std::allocator<int> >;

%template(matrix_i) std::vector<std::vector<int,std::allocator<int> >,std::allocator<std::vector<int,std::allocator<int> > > >;

%inline 
{
  typedef 
    std::vector<std::vector<int,std::allocator<int> >,
                std::allocator<std::vector<int,std::allocator<int> > > >
    imatrix;

  std::vector<int> vident(const std::vector<int,std::allocator<int> >& v)
  {
    return v;
  }

  imatrix mident(const imatrix& v)
  {
    return v;
  }
}


%template(DoubleVector) std::vector<double, std::allocator<double> >;

%inline %{
typedef float Real;
%}

namespace std {
  %template(RealVector) vector<Real, std::allocator<Real> >;
}

%inline %{

double average(std::vector<int, std::allocator<int> > v) {
    return std::accumulate(v.begin(),v.end(),0.0)/v.size();
}

std::vector<Real,std::allocator<Real> >
half(const std::vector<Real,std::allocator<Real> >& v) {
    std::vector<Real> w(v);
    for (unsigned int i=0; i<w.size(); i++)
        w[i] /= 2.0;
    return w;
}

%}

%template(IntPtrVector) std::vector<int *,std::allocator<int *> >;



