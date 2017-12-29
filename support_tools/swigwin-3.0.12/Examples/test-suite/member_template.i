%module member_template

%{
#ifdef max
#undef max
#endif
%}

  

%inline %{
template<class T> T max(T x, T y, T z) { return (x > y) ? x : y; }

template<class T> class Foo {
 public:
  template<class S> S max(S x, S y) { return (x > y) ? x : y; }
};

%}

%extend Foo {
  %template(maxi)   max<int>;
  %template(maxd)   max<double>;
};

%template(Fooint)    Foo<int>;
%template(Foodouble) Foo<double>;

