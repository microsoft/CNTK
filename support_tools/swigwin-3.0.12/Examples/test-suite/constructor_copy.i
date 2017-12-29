%module constructor_copy

%copyctor;
%nocopyctor Foo8;
%nocopyctor Bar<double>;

%inline %{

struct Foo1 {
  int x;

  Foo1(int _x = 2) : x(_x)
  {
  }
};

struct Foo2 {
  Foo2() { }
};

struct Foo3 {
  Foo3() { }
  Foo3(const Foo3& ) { }
};

struct Foo4 {
  Foo4() { }

protected:
  Foo4(const Foo4& ) { }
};


struct Foo4a {
  Foo4a() { }

private:
  Foo4a(const Foo4a& ) { }
};


struct Foo5 : Foo4 {
};

struct Foo6 : Foo4 {
  Foo6(const Foo6& f) : Foo4(f) { }
};

struct Foo7 : Foo5 {
};

struct Foo8 {
};

template <class T>
class Bar
{
public:
  int x;

  Bar(int _x = 0) : x(_x)
  {
  }
};
%}

%template(Bari) Bar<int>;
%template(Bard) Bar<double>;


#if defined(SWIGJAVA) || defined(SWIGCSHARP) || defined(SWIGUTL)


%include "std_vector.i"

#if defined(SWIGJAVA) || defined(SWIGCSHARP) || defined(SWIGPYTHON) || defined(SWIGR) || defined(SWIGOCTAVE) || defined(SWIGRUBY) || defined(SWIGJAVASCRIPT) || defined(SWIGSCILAB)
#define SWIG_GOOD_VECTOR
%ignore std::vector<Space::Flow>::vector(size_type);
%ignore std::vector<Space::Flow>::resize(size_type);
#endif

#if defined(SWIGTCL) || defined(SWIGPERL)
#define SWIG_GOOD_VECTOR
/* here, for languages with bad declaration */
%ignore std::vector<Space::Flow>::vector(unsigned int);
%ignore std::vector<Space::Flow>::resize(unsigned int);
#endif

%copyctor;

%ignore FlowFlow::FlowFlow;

%inline %{

namespace Space {
class Flow {
int x;
public:
 Flow(int i) : x(i) {}
};


class FlowFlow {
int x;
public:
 FlowFlow(int i) : x(i) {}
};

}

%}

%template (VectFlow) std::vector<Space::Flow>;

#endif


%rename(ABC_Libor_ModelUtils) ABC_Nam::ABC_Libor::ModelUtils;

%copyctor;
%inline %{
  namespace ABC_Nam {
    namespace ABC_Libor {
      struct ModelUtils {};

      template <class T>
      struct ModelUtils_T {};

    }
  }
%}

%template(ModelUtils_i) ABC_Nam::ABC_Libor::ModelUtils_T<int>;


%rename(Space1Space2_TotalReturnSwap) Space1::Space2::TotalReturnSwap;

%copyctor;

%inline %{
namespace Space1 {
  namespace Space2 {

    class TotalReturnSwap {
    public:
      TotalReturnSwap() {}
    };

    template <class T>
    class TotalReturnSwap_T {
    public:
      TotalReturnSwap_T() {}
    };

  }
}
%}

%template(Total_i) Space1::Space2::TotalReturnSwap_T<int>;

