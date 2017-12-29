%module template_matrix
%{
#include <vector>

  struct pop 
  {
  };  

%}

%include "std_vector.i"


%inline {
namespace simuPOP
{
  struct POP
  {
  };  

  template<class _POP1, class _POP2 = POP>
  class Operator
  {
    int x;
  };
}

}

%template(vectorop) std::vector< simuPOP::Operator<pop> >;



namespace simuPOP
{
  %template(baseOperator)        Operator<pop>;
}


#if 1

namespace std
{
%template(vectori)     vector<int>;
%template(matrixi)     vector< vector<int> >;
%template(cubei)       vector< vector< vector<int> > >;
}



%inline %{
std::vector<int>
passVector(const std::vector<int>& a)
{
  return a;
}

std::vector< std::vector<int> >
passMatrix(const  std::vector< std::vector<int> >& a)
{
  return a;
}

std::vector< std::vector< std::vector<int> > >
passCube(const  std::vector< std::vector< std::vector<int> > >& a)
{
  return a;
}

%}

#endif
