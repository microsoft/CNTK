%module li_std_vector

%include "std_vector.i"
%include "std_string.i"

%{
#include <algorithm>
#include <functional>
#include <numeric>
%}

namespace std {
    %template(IntVector) vector<int>;
}

%template(BoolVector) std::vector<bool>;
%template(CharVector) std::vector<char>;
%template(ShortVector) std::vector<short>;
%template(LongVector) std::vector<long>;
%template(UCharVector) std::vector<unsigned char>;
%template(UIntVector) std::vector<unsigned int>;
%template(UShortVector) std::vector<unsigned short>;
%template(ULongVector) std::vector<unsigned long>;
%template(DoubleVector) std::vector<double>;
%template(StringVector) std::vector<std::string>;


%inline %{
typedef float Real;
size_t typedef_test(std::vector<int>::size_type s) { return s; }
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

struct Struct {
  double num;
  Struct() : num(0.0) {}
  Struct(double d) : num(d) {}
};

struct Structure {
  double num;
  Structure() : num(0.0) {}
  Structure(double d) : num(d) {}
};

const std::vector<Real> & vecreal(const std::vector<Real> & vec) { return vec; }

const std::vector<int> & vecintptr(const std::vector<int> & vec) { return vec; }
const std::vector<int *> & vecintptr(const std::vector<int *> & vec) { return vec; }
const std::vector<const int *> & vecintconstptr(const std::vector<const int *> & vec) { return vec; }

const std::vector<Struct> & vecstruct(const std::vector<Struct> & vec) { return vec; }
const std::vector<Struct *> & vecstructptr(const std::vector<Struct *> & vec) { return vec; }
const std::vector<const Struct *> & vecstructconstptr(const std::vector<const Struct *> & vec) { return vec; }
%}

#if !defined(SWIGR)
%template(IntPtrVector) std::vector<int *>;
%template(IntConstPtrVector) std::vector<const int *>;
#endif
%template(StructVector) std::vector<Struct>;
%template(StructPtrVector) std::vector<Struct *>;
%template(StructConstPtrVector) std::vector<const Struct *>;

%inline {
  struct MyClass {};
  typedef MyClass *MyClassPtr;
  typedef std::vector<MyClassPtr> MyClassVector;
}
%template(MyClassPtrVector) std::vector<MyClassPtr>;

%inline {
  class RetsMetadata
  {
  public:
    MyClassVector GetAllResources(size_t n) const
    {
      return MyClassVector(n, 0);
    }
  };
}

#if defined(SWIGRUBY)
%template(LanguageVector) std::vector< swig::LANGUAGE_OBJ >;

%inline {
  std::vector< swig::LANGUAGE_OBJ > LanguageVector; 
}
#endif


// Test that the digraph <::aa::Holder> is not generated
%include <std_vector.i>

%inline %{
namespace aa {
  struct Holder {
    Holder(int n = 0) : number(n) {}
    int number;
  };
}
%}

#if !defined(SWIGOCTAVE)
// To fix: something different in Octave is preventing this from working
%template(VectorTest) std::vector< ::aa::Holder >;

%inline %{
std::vector< ::aa::Holder > vec1(std::vector< ::aa::Holder > x) { return x; }
%}
#endif

// exercising vectors of strings
%inline %{
std::vector<std::string> RevStringVec (const std::vector<std::string> &In)
  {
    std::vector<std::string> result(In);
    std::reverse(result.begin(), result.end());
    return(result);
  }
%}
