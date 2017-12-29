%module nested_template_base

%inline %{
  template <class T> class OuterT {
    public:
      T outer(T t) { return t; }
  };
%}
 
// The %template goes after OuterT and before OuterC as OuterC::InnerC's base is handled inside OuterC
%template(OuterTInnerS) OuterT<OuterC::InnerS>;

#if !defined(SWIGCSHARP) && !defined(SWIGJAVA)
%feature("flatnested") OuterC::InnerS;
%feature("flatnested") OuterC::InnerC;
#endif


%inline %{
  class OuterC {
  public:
    struct InnerS;
    class InnerC;
  };
 
  struct OuterC::InnerS {
    int val;
    InnerS(int val = 0) : val(val) {}
  };


  class OuterC::InnerC : public OuterT<InnerS> {
  public:
    OuterT<InnerS>& innerc() {
      return *this;
    }
  }; 
%}
