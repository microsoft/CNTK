%module li_std_pair

%include "std_pair.i"

namespace std {
  %template(IntPair)   pair<int, int>;
}

%inline %{

/* Test the "out" typemap for pair<T, U> */
std::pair<int, int> makeIntPair(int a, int b) {
    return std::make_pair(a, b);
}

/**
 * There is no "out" typemap for a pointer to a pair, so
 * this should return a wrapped instance of a std::pair
 * instead of the native "array" type for the target language.
 */
std::pair<int, int> * makeIntPairPtr(int a, int b) {
    static std::pair<int, int> p = std::make_pair(a, b);
    return &p;
}

/**
 * There is no "out" typemap for a non-const reference to a pair, so
 * this should return a wrapped instance of a std::pair instead of
 * the native "array" type for the target language.
 */
std::pair<int, int>& makeIntPairRef(int a, int b) {
    static std::pair<int, int> p = std::make_pair(a, b);
    return p;
}

/**
 * There is no "out" typemap for a const reference to a pair, so
 * this should return a wrapped instance of a std::pair
 * instead of the native "array" type for the target language.
 */
const std::pair<int, int> & makeIntPairConstRef(int a, int b) {
    static std::pair<int, int> p = std::make_pair(a, b);
    return p;
}

/* Test the "in" typemap for pair<T, U> */
int product1(std::pair<int, int> p) {
    return p.first*p.second;
}

/* Test the "in" typemap for const pair<T, U>& */
int product2(const std::pair<int, int>& p) {
    return p.first*p.second;
}

/* Test the "in" typemap for const pair<T, U>* */
int product3(const std::pair<int, int> *p) {
    return p->first*p->second;
}

%}

// Test that the digraph <::aa::Holder> is not generated for stl containers
%include <std_pair.i>

%inline %{
namespace aa {
  struct Holder {
    Holder(int n = 0) : number(n) {}
    int number;
  };
}
%}

%template(PairTest) std::pair< ::aa::Holder, int >;

%inline %{
std::pair< ::aa::Holder, int > pair1(std::pair< ::aa::Holder, int > x) { return x; }
%}
