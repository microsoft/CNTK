%module li_std_pair_extra

//
// activate the automatic comparison methods generation (==,!=,...) 
//

%{
#include <algorithm>                   // for std::swap
%}


%include std_pair.i
%include std_string.i
%include std_complex.i

%inline
%{
  struct A 
  {
    int val;
    
    A(int v = 0): val(v)
    {
    }
    
  };
  struct B
  {
  };
%}

%std_comp_methods(std::pair<std::string, int>);

namespace std {
  %template(CIntPair) pair<const int, const int>;
  %template() pair<double, double>;
  %template(ShortPair) pair<short, short>;

  %template(IntPair) pair<int, int>;
  %extend pair<int, int>
  {
    %template(pair) pair<short,short>;
  }
  
  

  %template(SIPair) pair<std::string, int>;
  %template(CIPair) pair<std::complex<double>, int>;
  %template(SIIPair) pair<std::pair<std::string, int>, int>;
  %template(AIntPair) pair<A, int>;

  %template(CCIntPair) pair<const A, const pair<int, int> >;

  %template(ABPair) pair<A, B>;
  %template(IntAPair) pair<int, A>;

  %template(pairP1) pair<int, A*>;
  %template(pairP2) pair<A*, int>;
  %template(pairP3) pair<A*, A*>;
  %template(pairP4) pair<int, int*>;
  %template(pairP5) pair<int*, int>;
  %template(pairP6) pair<int*, int*>;
  
}
%std_comp_methods(std::pair<std::pair<std::string, int>, int>);

%apply std::pair<int,int> *INOUT {std::pair<int,int> *INOUT2};

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

std::pair<int, int> 
  p_ident(std::pair<int, int> p, const std::pair<int, int>& q) {
  return p;
}


std::pair<int, A*> 
p_identa(const std::pair<int, A*>& p) {
  return p;
}

void
d_inout(double *INOUT) {
  *INOUT += *INOUT;
}

void
d_inout(int *INOUT) {
  *INOUT += *INOUT;
}

int
d_inout2(double *INOUT) {
  *INOUT += *INOUT;
  return 1;
}

void
p_inout(std::pair<int, int> *INOUT) {
  std::swap(INOUT->first, INOUT->second);
}

int
p_inout2(std::pair<int, int> *INOUT) {
  std::swap(INOUT->first, INOUT->second);
  return 1;
}

void
  p_inout3(std::pair<int,int> *INOUT, std::pair<int,int> *INOUT2) {
  std::swap(*INOUT, *INOUT2);
} 

void
p_inoutd(std::pair<double, double> *INOUT) {
  std::swap(INOUT->first, INOUT->second);
}

std::string
  s_ident(const std::string& s) {
  return s;
}

#if 0
std::pair<char, char> 
  p_ident(std::pair<char, char> p, const std::pair<char, char>& q) {
  return p;
}

/* Test the "in" typemap for const pair<T, U>* */
std::pair<A, B> 
  p_ident(std::pair<A, B> p, const std::pair<A, B>& q) {
  return q;
}

/* Test the "in" typemap for const pair<T, U>* */
std::pair<int, A> 
  p_ident(std::pair<int, A> p, const std::pair<int, A>& q) {
  return p;
}


std::pair<int, int> 
  p_ident(std::pair<int, int> p, const std::pair<A, int>& q) {
  return p;
}

std::pair<int, int> 
  p_ident(std::pair<int, int> p, const std::pair<A, B>& q) {
  return p;
}



#endif
%}


namespace std
{
  %template(paircA1) pair<const int, A*>;
  %template(paircA2) pair<const int, const A*>;
  %template(pairiiA) pair<int,pair<int, A*> >;
}

