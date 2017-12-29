/* File : example.i */
%module swigexample

%feature("autodoc", 1);

%inline %{
// From B. Strousjoup, "The C++ Programming Language, Third Edition", p. 514
template<class T> class Sum {
   T res;
public:
   Sum(T i = 0) : res(i) { }
   void operator() (T x) { res += x; }
   T result() const { return res; }
};

%}

// Rename the application operator to __call__ for python.
// Note: this is normally automatic, but if you had to do it yourself
// you would use this directive:
//
// %rename(__call__) *::operator();

// Instantiate a few versions
%template(intSum) Sum<int>;
%template(doubleSum) Sum<double>;
