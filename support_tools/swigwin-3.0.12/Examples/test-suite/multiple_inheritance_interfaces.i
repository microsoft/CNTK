%module  multiple_inheritance_interfaces

%warnfilter(SWIGWARN_RUBY_MULTIPLE_INHERITANCE,
	    SWIGWARN_D_MULTIPLE_INHERITANCE,
	    SWIGWARN_PHP_MULTIPLE_INHERITANCE); /* languages not supporting multiple inheritance or %interface */

#if defined(SWIGJAVA) || defined(SWIGCSHARP)
%include "swiginterface.i"
%interface_custom("A", "IA", IA)
%interface_custom("B", "IB", IB)
%interface_custom("%(strip:[I])s", "I%s", IC) // same as %interface_custom("C", "IC", IC)
#endif

%inline %{
struct IA {
  virtual void ia() {};
  virtual void ia(const char *s, bool b = true) {}
  virtual void ia(int i) {}
  virtual ~IA() {}
};
struct IB { virtual ~IB() {} virtual void ib() {} };
struct IC : IA, IB {};
struct D : IC {};
struct E : D {};
%}

#if defined(SWIGJAVA) || defined(SWIGCSHARP)
%interface_custom("J", "IJ", IJ)
%interface_custom("K", "IK", IK)
%interface_custom("L", "IL", IL)
#endif
%inline %{
struct IJ { virtual ~IJ() {}; virtual void ij() {} };
struct IK : IJ {};
struct IL : IK {};
struct M : IL {};
%}

#if defined(SWIGJAVA) || defined(SWIGCSHARP)
%interface_custom("Q", "IQ", IQ)
#endif
%inline %{
struct P  { virtual ~P() {}  virtual void p()  {} };
struct IQ { virtual ~IQ() {} virtual void iq() {} };
struct R : IQ, P {};
struct S : P, IQ {};
struct T : IQ {};
struct U : R {};
struct V : S {};
struct W : T {};
%}

#if defined(SWIGJAVA) || defined(SWIGCSHARP)
%interface_impl(BaseOverloaded);
#endif
%inline %{
struct BaseOverloaded {
  typedef P PTypedef;
  virtual ~BaseOverloaded() {}
  virtual void identical_overload(int i, const PTypedef &pp = PTypedef()) {}
};

struct DerivedOverloaded : public BaseOverloaded {
  virtual void identical_overload(int i, const PTypedef &p = PTypedef()) {}
};
%}
