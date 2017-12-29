%module("templatereduce") li_std_map
%feature("trackobjects");

%inline %{
namespace another {
struct map {
  int val;
  map(int x) : val(x) {}
};
}
%}

%include "std_pair.i"
%include "std_map.i"
%include "std_string.i"

// Declare some maps to play around with
%template(IntIntMap) std::map<int, int>;
%template(StringIntMap) std::map<std::string, int>;

%ignore Struct::operator<;
%ignore Struct::operator==;

// Add an inline function to test
%inline %{

double valueAverage(std::map<std::string, int> m) {
  if (m.size() == 0) {
    return 0.0;
  }
    
  double a = 0.0;
  for (std::map<std::string, int>::iterator i = m.begin(); i != m.end(); i++) {
    a += i->second;
  }
    
  return a / m.size();
}
    
std::string stringifyKeys(std::map<std::string, int> m) {
  std::string a;
  for (std::map<std::string, int>::iterator i = m.begin(); i != m.end(); i++) {
    a += " " + i->first;
  }
  return a;
}

struct Struct {
  double num;
  Struct() : num(0.0) {}
  Struct(double d) : num(d) {}
  bool operator<(const Struct &other) const { return num < other.num; }
  bool operator==(const Struct &other) const { return num == other.num; }
};

%}

//#if !defined(SWIGR)

// Test out some maps with pointer types
%template(IntIntPtrMap) std::map<int, int *>;
%template(IntConstIntPtrMap) std::map<int, const int *>;

//#endif


// Test out some maps with non-basic types and non-basic pointer types
%template(IntStructMap) std::map<int, Struct>;
%template(IntStructPtrMap) std::map<int, Struct *>;
%template(IntStructConstPtrMap) std::map<int, const Struct *>;
%template(StructPtrIntMap) std::map<Struct *, int>;

// Test out a non-specialized map
%template(StructIntMap) std::map<Struct, int>;

// Additional map definitions for Ruby, Python and Octave tests
%inline %{
  struct A{
    int val;
    
    A(int v = 0): val(v) {
    }
  };
%}

namespace std {
  %template(pairii) pair<int, int>;
  %template(pairAA) pair<int, A>;
  %template(pairA) pair<int, A*>;
  %template(mapA) map<int, A*>;

  %template(paircA1) pair<const int, A*>;
  %template(paircA2) pair<const int, const A*>;
  %template(pairiiA) pair<int,pair<int, A*> >;
  %template(pairiiAc) pair<int,const pair<int, A*> >;


#ifdef SWIGRUBY
  %template() pair< swig::LANGUAGE_OBJ, swig::LANGUAGE_OBJ >;
  %template(LanguageMap) map< swig::LANGUAGE_OBJ, swig::LANGUAGE_OBJ >;
#endif

#ifdef SWIGPYTHON
  %template() pair<swig::SwigPtr_PyObject, swig::SwigPtr_PyObject>;
  %template(pymap) map<swig::SwigPtr_PyObject, swig::SwigPtr_PyObject>;
#endif
  
}

%inline {
  std::pair<int, A*> p_identa(std::pair<int, A*> p) {
    return p;
  }

  std::map<int, A*> m_identa(const std::map<int,A*>& v) {
    return v;
  }
}


