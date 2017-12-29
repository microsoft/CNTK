%module li_std_list

%include "std_list.i"

%{
#include <algorithm>
#include <functional>
#include <numeric>
%}

namespace std {
    %template(IntList) list<int>;
}

%template(DoubleList) std::list<double>;

%inline %{
typedef float Real;
%}

namespace std {
    %template(RealList) list<Real>;
}

%inline %{

double average(std::list<int> v) {
    return std::accumulate(v.begin(),v.end(),0.0)/v.size();
}


void halve_in_place(std::list<double>& v) {
    std::transform(v.begin(),v.end(),v.begin(),
                   std::bind2nd(std::divides<double>(),2.0));
}

struct Struct {
  double num;
  Struct() : num(0.0) {}
  Struct(double d) : num(d) {}
//  bool operator==(const Struct &other) { return (num == other.num); }
};
%}



