// Tests the use of the %template directive with fully
// qualified scope names

%module template_ns

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) std::my_pair<int, int>;       /* Ruby, wrong class name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) std::my_pair<double, double>; /* Ruby, wrong class name */

%ignore std::my_pair::my_pair();

%inline %{
namespace std
{
template <class _T1, class _T2>
struct my_pair {
  typedef _T1 first_type; 
  typedef _T2 second_type;

  _T1 first;              
  _T2 second;             
  my_pair() : first(_T1()), second(_T2()) {}
  my_pair(const _T1& __a, const _T2& __b) : first(__a), second(__b) {}
  template <class _U1, class _U2>
  my_pair(const my_pair<_U1, _U2>& __p) : first(__p.first), second(__p.second) {}
};
}
%}

// Add copy constructor
%extend std::my_pair {
   %template(pair) my_pair<_T1,_T2>;
};

%template(pairii) std::my_pair<int,int>;
%template(pairdd) std::my_pair<double,double>;
