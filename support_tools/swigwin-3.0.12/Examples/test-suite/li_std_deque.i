%module li_std_deque

%include "std_deque.i"

%{
#include <algorithm>
#include <functional>
#include <numeric>
%}

namespace std {
    %template(IntDeque) deque<int>;
}

%template(DoubleDeque) std::deque<double>;

%inline %{
typedef float Real;
%}

namespace std {
    %template(RealDeque) deque<Real>;
}

%inline %{

double average(std::deque<int> v) {
    return std::accumulate(v.begin(),v.end(),0.0)/v.size();
}

std::deque<float> half(const std::deque<float>& v) {
    std::deque<float> w(v);
    for (unsigned int i=0; i<w.size(); i++)
        w[i] /= 2.0;
    return w;
}

void halve_in_place(std::deque<double>& v) {
    std::transform(v.begin(),v.end(),v.begin(),
                   std::bind2nd(std::divides<double>(),2.0));
}

%}




