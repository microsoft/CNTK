// This testcase is almost identical to template_partial_specialization but uses typedefs for %template

%module template_partial_specialization_typedef

%inline %{
namespace TypeDef {
  typedef double Double;
  typedef int * IntPtr;
  typedef double * DoublePtr;
  typedef double & DoubleRef;
  typedef const double & ConstDoubleRef;
  typedef double * const & DoublePtrConstRef;

  typedef int Int;
  typedef int * const & IntPtrConstRef;
  typedef int ** IntPtrPtr;
  typedef float Float;
  typedef float * FloatPtr;
  typedef float ** FloatPtrPtr;
  typedef float *** FloatPtrPtrPtr;

  typedef bool * BoolPtr;
  typedef char * CharPtr;
  typedef short * ShortPtr;
  typedef long * LongPtr;
  typedef unsigned int ** UnsignedIntPtrPtr;
  typedef unsigned int *** UnsignedIntPtrPtrPtr;
  typedef const unsigned int ** ConstUnsignedIntPtr;
  typedef const unsigned int *** ConstUnsignedIntPtrPtr;
}
namespace One {
  template <typename T> struct OneParm                  { void a() {} };
  template <typename T> struct OneParm<T *>             { void b() {} };
  template <typename T> struct OneParm<T &>             { void c() {} };
  template <typename T> struct OneParm<T const &>       { void d() {} };
  template <typename T> struct OneParm<T * const &>     { void e() {} };

  template <>           struct OneParm<int>             { void f() {} };
  template <>           struct OneParm<int * const &>   { void g() {} };
  template <>           struct OneParm<int **>          { void h() {} };

  template <>           struct OneParm<float>           { void i() {} };
  template <>           struct OneParm<float *>         { void j() {} };
  template <>           struct OneParm<float **>        { void k() {} };
  template <>           struct OneParm<float ***>       { void l() {} };
}
%}

// partial specializations
%template(A) One::OneParm<TypeDef::Double>;
%template(B) One::OneParm<TypeDef::DoublePtr>;
%template(C) One::OneParm<TypeDef::DoubleRef>;
%template(D) One::OneParm<TypeDef::ConstDoubleRef>;
%template(E) One::OneParm<TypeDef::DoublePtrConstRef>;

// explicit specializations
%template(F) One::OneParm<TypeDef::Int>;
%template(G) One::OneParm<TypeDef::IntPtrConstRef>;
%template(H) One::OneParm<TypeDef::IntPtrPtr>;

// %template scope explicit specializations
namespace ONE {
  %template(I) One::OneParm<TypeDef::Float>;
  %template(J) ::One::OneParm<TypeDef::FloatPtr>;
}
%template(K) ::One::OneParm<TypeDef::FloatPtrPtr>;
namespace One {
  %template(L) OneParm<TypeDef::FloatPtrPtrPtr>;
}

// %template scope partial specializations
namespace ONE {
  %template(BB) One::OneParm<TypeDef::BoolPtr>;
  %template(BBB) ::One::OneParm<TypeDef::CharPtr>;
}
%template(BBBB) ::One::OneParm<TypeDef::ShortPtr>;
namespace One {
  %template(BBBBB) OneParm<TypeDef::LongPtr>;
}

// non-exact match
%template(B1) One::OneParm<TypeDef::UnsignedIntPtrPtr>;
%template(B2) One::OneParm<TypeDef::UnsignedIntPtrPtrPtr>;
%template(B3) One::OneParm<TypeDef::ConstUnsignedIntPtr>;
%template(B4) One::OneParm<TypeDef::ConstUnsignedIntPtrPtr>;


// Two parameter specialization tests
%inline %{
struct Concrete {};
namespace Two {
  template <typename T1, typename T2> struct TwoParm                          { void a() {} };
  template <typename T1, typename T2> struct TwoParm<T1 *, T2 *>              { void b() {} };
  template <typename T1, typename T2> struct TwoParm<T1 *, const T2 *>        { void c() {} };
  template <typename T1, typename T2> struct TwoParm<const T1 *, const T2 *>  { void d() {} };
  template <typename T1>              struct TwoParm<T1 *, int *>             { void e() {} };
  template <typename T1>              struct TwoParm<T1, int>                 { void f() {} };
  template <>                         struct TwoParm<int *, const int *>      { void g() {} };
}
%}

%inline %{
namespace TypeDef {
  typedef const double * ConstDoublePtr;
  typedef const int * ConstIntPtr;
  typedef int * IntPtr;
  typedef Concrete * ConcretePtr;
  typedef const Concrete * ConstConcretePtr;
  typedef void * VoidPtr;
}
%}
namespace Two {
  %template(A_) TwoParm<TypeDef::Double, TypeDef::Double>;
  %template(B_) TwoParm<TypeDef::DoublePtr, TypeDef::DoublePtr>;
  %template(C_) TwoParm<TypeDef::DoublePtr, TypeDef::ConstDoublePtr>;
  %template(D_) TwoParm<TypeDef::ConstIntPtr, TypeDef::ConstIntPtr>;
  %template(E_) TwoParm<TypeDef::IntPtr, TypeDef::IntPtr>;
  %template(F_) TwoParm<TypeDef::IntPtr, TypeDef::Int>;
  %template(G_) TwoParm<TypeDef::IntPtr, TypeDef::ConstIntPtr>;

  %template(C1_) TwoParm<TypeDef::ConcretePtr, TypeDef::ConstConcretePtr>;
  %template(C2_) TwoParm<TypeDef::IntPtr, TypeDef::ConstConcretePtr>;
}

%template(C3_) Two::TwoParm<TypeDef::DoublePtr, TypeDef::ConstConcretePtr>;
%template(C4_) ::Two::TwoParm<TypeDef::VoidPtr, TypeDef::ConstConcretePtr>;
%template(B1_) ::Two::TwoParm<TypeDef::CharPtr, TypeDef::ConcretePtr>;
%template(E1_) Two::TwoParm<TypeDef::ConstIntPtr, TypeDef::IntPtr>;
%template(E2_) Two::TwoParm<TypeDef::IntPtrPtr, TypeDef::IntPtr>;

