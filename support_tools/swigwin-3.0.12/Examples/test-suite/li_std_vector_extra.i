%module li_std_vector_extra

%warnfilter(509) overloaded1;
%warnfilter(509) overloaded2;

%include "std_string.i"
%include "std_vector.i"
%include "cpointer.i"
%include "carrays.i"

%{
#include <algorithm>
#include <functional>
#include <numeric>


#if defined(__clang__)
// Suppress:
// warning: destination for this 'memset' call is a pointer to dynamic class
//       'Test::B'; vtable pointer will be overwritten [-Wdynamic-class-memaccess]
//         memset(v_def,0,sizeof(Type));
// Better generated code is probably needed though
#pragma clang diagnostic ignored "-Wdynamic-class-memaccess"
#endif

%}

namespace std {
    %template() vector<short>;
    %template(IntVector) vector<int>;
    %template(BoolVector) vector<bool>;
    %template() vector<string>;
}

%template(DoubleVector) std::vector<double>;


%template(sizeVector) std::vector<size_t>;
%{
  template <class T>
  struct Param
  {
    T val;

    Param(T v = 0): val(v) {
    }
    
    operator T() const { return val; }
  };
%}
specialize_std_vector(Param<int>,PyInt_Check,PyInt_AsLong,PyInt_FromLong);
%template(PIntVector) std::vector<Param<int> >;

%inline %{
typedef float Real;
%}

namespace std {
    %template(RealVector) vector<Real>;
}

%inline %{

double average(std::vector<int> v) {
    return std::accumulate(v.begin(),v.end(),0.0)/v.size();
}

std::vector<Real> half(const std::vector<Real>& v) {
    std::vector<Real> w(v);
    for (std::vector<Real>::size_type i=0; i<w.size(); i++)
        w[i] /= 2.0;
    return w;
}

void halve_in_place(std::vector<double>& v) {
    std::transform(v.begin(),v.end(),v.begin(),
                   std::bind2nd(std::divides<double>(),2.0));
}

%}

%template(IntPtrVector) std::vector<int *>;



//
//
%{
#include <iostream>
%}

%inline %{
  
namespace Test {
struct A {
    virtual ~A() {}    
    virtual int f(const int i) const = 0;
};

struct B : public A {
  int val;
  
  B(int i = 0) : val(i)
  {
  }
  
  int f(const int i) const { return i + val; }
};


int vecAptr(const std::vector<A*>& v) {
    return v[0]->f(1);
}

} 

std::vector<short> halfs(const std::vector<short>& v) {
    std::vector<short> w(v);
    for (std::vector<short>::size_type i=0; i<w.size(); i++)
        w[i] /= 2;
    return w;
}


std::vector<std::string>  vecStr(std::vector<std::string> v) {
  v[0] += v[1];
  return v;
}

%}
%template(VecB) std::vector<Test::B>; 
%template(VecA) std::vector<Test::A*>; 

%pointer_class(int,PtrInt)
%array_functions(int,ArrInt)

%inline %{
  int *makeIntPtr(int v) { return new int(v); }
  const short *makeConstShortPtr(int v) { return new short(v); }
  double *makeDoublePtr(double v) { return new double(v); }
  int extractInt(int *p) { return *p; }
  short extractConstShort(const short *p) { return *p; }
%}

%template(pyvector) std::vector<swig::SwigPtr_PyObject>; 

namespace std {
   %template(ConstShortPtrVector) vector<const short *>;
}

%inline %{
std::string overloaded1(std::vector<double> vi) { return "vector<double>"; }
std::string overloaded1(std::vector<int> vi) { return "vector<int>"; }
std::string overloaded2(std::vector<int> vi) { return "vector<int>"; }
std::string overloaded2(std::vector<double> vi) { return "vector<double>"; }
std::string overloaded3(std::vector<int> *vi) { return "vector<int> *"; }
std::string overloaded3(int i) { return "int"; }
%}

