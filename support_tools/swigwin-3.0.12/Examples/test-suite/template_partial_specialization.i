%module template_partial_specialization

%inline %{
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
%template(A) One::OneParm<double>;
%template(B) One::OneParm<double *>;
%template(C) One::OneParm<double &>;
%template(D) One::OneParm<const double &>;
%template(E) One::OneParm<double * const &>;

// explicit specializations
%template(F) One::OneParm<int>;
%template(G) One::OneParm<int * const &>;
%template(H) One::OneParm<int **>;

// %template scope explicit specializations
namespace ONE {
  %template(I) One::OneParm<float>;
  %template(J) ::One::OneParm<float *>;
}
%template(K) ::One::OneParm<float **>;
namespace One {
  %template(L) OneParm<float ***>;
}

// %template scope partial specializations
namespace ONE {
  %template(BB) One::OneParm<bool *>;
  %template(BBB) ::One::OneParm<char *>;
}
%template(BBBB) ::One::OneParm<short *>;
namespace One {
  %template(BBBBB) OneParm<long *>;
}

// non-exact match
%template(B1) One::OneParm<unsigned int **>;
%template(B2) One::OneParm<unsigned int ***>;
%template(B3) One::OneParm<const unsigned int *>;
%template(B4) One::OneParm<const unsigned int **>;


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
  template <>                         struct TwoParm<Concrete, Concrete *>    { void h() {} };
}
%}

namespace Two {
  %template(A_) TwoParm<double, double>;
  %template(B_) TwoParm<double *, double *>;
  %template(C_) TwoParm<double *, const double *>;
  %template(D_) TwoParm<const int *, const int *>;
  %template(E_) TwoParm<int *, int *>;
  %template(F_) TwoParm<int *, int>;
  %template(G_) TwoParm<int *, const int *>;

  %template(C1_) TwoParm<Concrete *, const Concrete *>;
  %template(C2_) TwoParm<int *, const ::Concrete *>;
}

%template(C3_) Two::TwoParm<double *, const ::Concrete *>;
%template(C4_) ::Two::TwoParm<void *, const ::Concrete *>;
%template(B1_) ::Two::TwoParm<char *, ::Concrete *>;
%template(E1_) Two::TwoParm<const int *, int *>;
%template(E2_) Two::TwoParm<int **, int *>;
%template(H_) Two::TwoParm< ::Concrete, ::Concrete * >;


// Many template parameters
%inline %{
template <typename T1, typename T2, typename T3, typename T4, typename T5> struct FiveParm                               { void a() {} };
template <typename T1>                                                     struct FiveParm<T1, int, int, double, short>  { void b() {} };
%}

%template(FiveParm1) FiveParm<bool, int, int, double, short>;

%inline %{
template <typename T, int N = 0, int M = 0> struct ThreeParm;
template <typename T, int N, int M>         struct ThreeParm          { void a1() {} };
template <typename T>                       struct ThreeParm<T, 0, 0> { void a2() {} };
template <typename T, int N>                struct ThreeParm<T, N, N> { void a3() {} };
%}

%template(ThreeParmInt) ThreeParm<int, 0, 0>;

#if 0
// TODO fix:
%inline %{
//namespace S {
  template<typename T> struct X      { void a() {} };
  template<typename T> struct X<T *> { void b() {} };
//  template<>           struct X<int *> { void c() {} };
//}
%}

namespace AA {  // thinks X is in AA namespace
  %template(X2) X<int *>;
};
#endif

#if 0
namespace Space {
}
template<typename T> struct Vector {
#ifdef SWIG
  %template() Space::VectorHelper<T>;
#endif
  void gook(T i) {}
  void geeko(double d) {}
  void geeky(int d) {}
};
/*
template<typename T> struct Vector<T *> {
};
*/
//}
%}

%template(VectorIntPtr) Space::Vector<int *>; // should fail as Vector is in global namespace
// is this a regression - no fails in 1.3.40 too
// Note problem is removed by removing empty Space namespace!!
#endif
