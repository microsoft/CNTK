/* File : example.h -- stolen from the guile std_vector example */

#include <vector>
#include <algorithm>
#include <functional>
#include <numeric>

double average(std::vector<int> v) {
    return std::accumulate(v.begin(),v.end(),0.0)/v.size();
}

std::vector<double> half(const std::vector<double>& v) {
    std::vector<double> w(v);
    for (unsigned int i=0; i<w.size(); i++)
        w[i] /= 2.0;
    return w;
}

void halve_in_place(std::vector<double>& v) {
    // would you believe this is the same as the above?
    std::transform(v.begin(),v.end(),v.begin(),
                   std::bind2nd(std::divides<double>(),2.0));
}


