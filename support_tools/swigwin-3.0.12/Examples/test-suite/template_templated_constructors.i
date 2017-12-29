%module template_templated_constructors

%inline %{
namespace ConstructSpace {

class TConstructor1 {
public:
  template<typename T> TConstructor1(T val) {}
  ~TConstructor1() {}
};

class TConstructor2 {
public:
  TConstructor2() {}
  template<typename T> TConstructor2(T val) {}
  ~TConstructor2() {}
};

template<typename T> class TClass1 {
public:
  template<typename Y> TClass1(Y t) {}
};

template<typename T> class TClass2 {
public:
  TClass2() {}
  template<typename Y> TClass2(Y t) {}
};

}
%}

%extend ConstructSpace::TConstructor1 {
  %template(TConstructor1) TConstructor1<int>;
}

%template(TConstructor2) ConstructSpace::TConstructor2::TConstructor2<int>;

%template(TClass1Int) ConstructSpace::TClass1<int>;
%extend ConstructSpace::TClass1<int> {
  %template(TClass1Int) TClass1<double>;
}

%template(TClass2Int) ConstructSpace::TClass2<int>;
%extend ConstructSpace::TClass2<int> {
  %template(TClass2Int) TClass2<double>;
}
