/* This unit tests whether SWIG correctly parses the code and makes wrappers
   for the new C++11 extern templates (explicit template instantiation without
   using the translation unit).
*/
%module cpp11_template_explicit

#pragma SWIG nowarn=SWIGWARN_PARSE_EXPLICIT_TEMPLATE

%inline %{

template<typename T> struct Temper {
  T val;
};

class A {
public:
  int member;
  int memberFunction() { return 100; }
};

template class Temper<A>;
extern template class Temper<A>;

template class Temper<A*>;
extern template class Temper<A*>;

template class Temper<int>;
extern template class Temper<int>;
%}

%template(TemperInt) Temper<int>;
