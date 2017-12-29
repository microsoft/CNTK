%module ignore_template_constructor
%include std_vector.i

#if defined(SWIGJAVA) || defined(SWIGCSHARP) || defined(SWIGPYTHON) || defined(SWIGPERL) || defined(SWIGRUBY) 
#define SWIG_GOOD_VECTOR
%ignore std::vector<Flow>::vector(size_type);
%ignore std::vector<Flow>::resize(size_type);
#endif

#if defined(SWIGTCL) || defined(SWIGPERL)
#define SWIG_GOOD_VECTOR
/* here, for languages with bad declaration */
%ignore std::vector<Flow>::vector(unsigned int);
%ignore std::vector<Flow>::resize(unsigned int);
#endif

#if defined(SWIG_GOOD_VECTOR)
%inline %{
class Flow {
double x;
 Flow():x(0.0) {}
public:
 Flow(double d) : x(d) {}
};
%}

#else
/* here, for languages with bad typemaps */

%inline %{
class Flow {
double x;
public:
 Flow(): x(0.0) {}
 Flow(double d) : x(d) {}
};
%}

#endif

%template(VectFlow) std::vector<Flow>;

%inline %{
std::vector<Flow> inandout(std::vector<Flow> v) {
  return v;
}
%}
