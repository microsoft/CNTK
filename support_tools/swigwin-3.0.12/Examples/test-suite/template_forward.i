%module template_forward

%{
namespace foo {
template<class T> class bar { };
}
%}

namespace foo {
   template<class T> class bar;
};

%inline %{
namespace foo {
   double test1(const bar<double> &x) { return 0; }
   bar<double> test2() {
	return bar<double>();
   }
}
%}



%inline {
  // Forward declarations
  template<class RangeScalar, class DomainScalar = RangeScalar> class LinearOpBase;
  template<class Scalar>  class VectorBase;  
}


%inline {
  // Class Describable
  class Describable {
  public:
    void describe() {}
  };
  
  // Class LinearOpBase
  template<class RangeScalar, class DomainScalar> 
    class LinearOpBase : virtual public Describable {
    public:
      
    }; // end class LinearOpBase<RangeScalar,DomainScalar>
  
  // Class VectorBase
  template<class Scalar>
    class VectorBase : virtual public LinearOpBase<Scalar>
    {
    public:
      using LinearOpBase<Scalar>::describe;
    }; // end class VectorBase<Scalar>
  
}


%template (LinearOpBase_double)    LinearOpBase<double>;
%template (VectorBase_double)      VectorBase<double>;
%template (LinearOpBase_int)    LinearOpBase<int,int>;
%template (VectorBase_int)      VectorBase<int>;

// Template forward class declarations mixing class and typename without always naming the templated parameter name
%inline %{
template <class> class TClass1;
template <typename> class TClass2;
template <class, typename> class TClass3;
template <class, class, class> class TClass4;
template <typename, typename> class TClass5;
template <typename, class K = double> class TClass6;
template<typename, class K, class C = K> class TClass7;
%}

