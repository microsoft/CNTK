%module template_nested

#if !defined(SWIGCSHARP) && !defined(SWIGJAVA)
%feature ("flatnested");
#endif

// Test nested templates - that is template classes and template methods within a class.

namespace ns {
template <class T> struct ForwardTemplate;
}
%{
namespace ns {
  template <class T> struct ForwardTemplate {
    void tmethod(T t) {}
  };
}
%}

%inline %{

namespace ns {

  class NormalClass {
  public:
    NormalClass() {}
    ~NormalClass() {}
  };

  template <class T> struct NormalTemplate {
    void tmethod(T t) {}
  };
}
%}
%template(T_NormalTemplateNormalClass) ns::NormalTemplate<ns::NormalClass>;
%template(T_NormalTemplateInt) ns::NormalTemplate<int>;
%template(T_NormalTemplateDouble) ns::NormalTemplate<double>;

%inline %{
namespace ns {
  class OuterClass {
  public:
    template <class T> struct Inner1 {
      template <class U> struct SuperInner1 {
        void method1(U t) {}
      };
      template <class V> struct SuperInner2 {
        void method1(V t) {}
      };
      template <class W> void tmethod(W w) {}
      template <class X> void tmethodAgain(X x) {}
      template <class Y> struct SuperBase : public SuperInner1<Y> {
        void method1(Y y) {}
      };
    };
    Inner1<int> useInner1(const Inner1<int>& inner) { return inner; }

    template <class Z> void InnerTMethod(Z z) {}

    template <class T> class Inner2 : public NormalTemplate<T> {
    public:
      template <class U> class SuperInner1 {
      public:
        SuperInner1() {}
        void method1(U t) {}
      };
      template <class V> struct SuperInner2 {
        void method1(V t) {}
      };
      int embeddedVar;
      template <class X> void tmethod(X x) {}
      template <class Y> struct SuperBase : public SuperInner1<Y> {
        void method1(Y y) {}
      };
    };
    Inner2<int> useInner2(const Inner2<int>& inner) { return inner; }
    Inner2<NormalClass> useInner2Again(const Inner2<NormalClass>& inner) { return inner; }
#ifdef SWIG
    %template(T_OuterClassInner1Double) Inner1<double>;
#endif
    int iii;
  };
  struct ABC {
    ABC() {}
    ~ABC() {}
  };

  struct TemplateFuncs {
    template <class X> X templateMethod1(X x) { return x; }
    template <class X> X templateMethod2(X x) { return x; }
  };

  template <typename UU> struct OuterTemplate {
    template <typename VV> struct NestedInnerTemplate1 {
      template <typename Z> void NestedInnerInnerTMethod(Z z) {}
      void hohum() {}
    };
    template <typename W> void NestedInnerTMethod(UU u, W w) {}
    template <typename VV> struct NestedInnerTemplate2 {
      void hohum() {}
    };
    UU hohum(UU u) { return u; }
    template <typename VV> struct NestedInnerTemplate3 : public NestedInnerTemplate2<VV> {
      void hohum() {}
    };
    struct NestedStruct {
      NestedStruct() {}
      void hohum() {}
    };
    NestedInnerTemplate1<short> useNestedInnerTemplate1(const NestedInnerTemplate1<short>& inner) { return inner; }
    NestedInnerTemplate2<short> useNestedInnerTemplate2(const NestedInnerTemplate2<short>& inner) { return inner; }
    NestedInnerTemplate3<short> useNestedInnerTemplate3(const NestedInnerTemplate3<short>& inner) { return inner; }
    NestedStruct useNestedStruct(const NestedStruct& inner) { return inner; }
  };
}
%}

#if !defined(SWIGSCILAB)
%extend ns::OuterClass {
  %template(T_OuterClassInner2Double) Inner2<double>;
}

%template(T_OuterTMethodNormalClass) ns::OuterClass::InnerTMethod<ns::NormalClass>;
%template(T_TemplateFuncs1Int) ns::TemplateFuncs::templateMethod1<int>;
%template(T_TemplateFuncs2Double) ns::TemplateFuncs::templateMethod2<double>;
%template(T_NestedOuterTemplateDouble) ns::OuterTemplate<double>;
%template(T_OuterClassInner1Int) ns::OuterClass::Inner1<int>;
%template(T_OuterClassInner2NormalClass) ns::OuterClass::Inner2<ns::NormalClass>;
%template(T_OuterClassInner2Int) ns::OuterClass::Inner2<int>;

#else
%extend ns::OuterClass {
  %template(T_OutClsIn2Dbl) Inner2<double>;
}

%template(T_OutClsIn1Int) ns::OuterClass::Inner1<int>;
%template(T_OutClsIn2NormCls) ns::OuterClass::Inner2<ns::NormalClass>;
%template(T_OutClsIn2Int) ns::OuterClass::Inner2<int>;

#endif
