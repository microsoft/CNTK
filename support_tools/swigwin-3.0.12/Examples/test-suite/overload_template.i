%module overload_template

#ifdef SWIGLUA
// lua only has one numeric type, so most of the overloads shadow each other creating warnings
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) foo;
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) maximum;
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) specialization;
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) overload;
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) space::nsoverload;
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) fooT;
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) barT;
#endif

%inline %{

int foo() {
  return 3;
}

template <class T>
  int foo(T x) {
     return (int)x;
  }

template<class T>
  T maximum(T a, T b) { return  (a > b) ? a : b; }
%}                                     


%template(foo) foo<int>;
%template(foo) foo<double>;

%template(maximum) maximum<int>;
%template(maximum) maximum<double>;

// Mix template overloading with plain function overload
// Mix 1
%inline %{
  int mix1(const char* msg) { return 101; }
  template<typename T> int mix1(T t, const T& tt) { return 102; }
  template<typename T> int mix1(T t) { return 103; }
%}
%template(mix1) mix1<double>;

// Mix 2
%inline %{
  template<typename T> int mix2(T t, const T& tt) { return 102; }
  int mix2(const char* msg) { return 101; }
  template<typename T> int mix2(T t) { return 103; }
%}
%template(mix2) mix2<double>;

// Mix 3
%inline %{
  template<typename T> int mix3(T t, const T& tt) { return 102; }
  template<typename T> int mix3(T t) { return 103; }
  int mix3(const char* msg) { return 101; }
%}
%template(mix3) mix3<double>;


// overloaded by number of templated parameters
// Combination 1
%inline %{
template<typename T> int overtparams1(T t) { return 10; }
template<typename T, typename U> int overtparams1(T t, U u) { return 20; }
%}

%template(overtparams1) overtparams1<int>;
%template(overtparams1) overtparams1<double, int>;


// Combination 2
%inline %{
template<typename T> int overtparams2(T t) { return 30; }
template<typename T, typename U> int overtparams2(T t, U u) { return 40; }
%}

%template(overtparams2) overtparams2<double, int>;


// Combination 3
%inline %{
template<typename T> int overloaded(T t) { return 50; }
int overloaded() { return 60; }
template<typename T, typename U> int overloaded(T t, U u) { return 70; }
%}

%template(overloaded) overloaded<double, int>;

// Combination 4
%inline %{
int overloadedagain(const char* msg) { return 80; }
template<typename T> int overloadedagain() { return 90; }
template<typename T, typename U> int overloadedagain(T t, U u) { return 100; }
%}

%template(overloadedagain) overloadedagain<double>;

// simple specialization
%inline %{
template<typename T> void xyz() {}
template<> void xyz<double>() {}
void xyz() {}
%}

// We can have xyz(); xyz<double>(); xyz<int>(); in C++, but can't have this type of overloading in target language, so we need to do some renaming
%template(xyz_double) xyz<double>;
%template(xyz_int) xyz<int>;


// specializations
%inline %{
template<typename T> int specialization(T t) { return 200; }
template<typename T, typename U> int specialization(T t, U u) { return 201; }
template<> int specialization(int t) { return 202; }
template<> int specialization<double>(double t) { return 203; }
template<> int specialization(int t, int u) { return 204; }
template<> int specialization<double,double>(double t, double u) { return 205; }
%}

%template(specialization) specialization<int>;
%template(specialization) specialization<double>;
%template(specialization) specialization<int, int>;
%template(specialization) specialization<double, double>;
%template(specialization) specialization<const char *, const char *>;


// a bit of everything
%inline %{
int overload(const char *c) { return 0; }
template<typename T> int overload(T t) { return 10; }
template<typename T> int overload(T t, const T &tref) { return 20; }
template<typename T> int overload(T t, const char *c) { return 30; }
template<> int overload<double>(double t, const char *c) { return 40; }
int overload() { return 50; }

class Klass {};
%}

%template(overload) overload<int>;
%template(overload) overload<Klass>;
%template(overload) overload<double>;


// everything put in a namespace
%inline %{
namespace space {
  int nsoverload(const char *c) { return 1000; }
  template<typename T> int nsoverload(T t) { return 1010; }
  template<typename T> int nsoverload(T t, const T &tref) { return 1020; }
  template<typename T> int nsoverload(T t, const char *c) { return 1030; }
  template<> int nsoverload<double>(double t, const char *c) { return 1040; }
  int nsoverload() { return 1050; }
}
%}

%template(nsoverload) space::nsoverload<int>;
%template(nsoverload) space::nsoverload<Klass>;
%template(nsoverload) space::nsoverload<double>;


%inline %{
  namespace space 
  {
    template <class T>
    struct Foo 
    {
      void bar(T t1) { }
      void bar(T t1, T t2) { }
      void bar(int a, int b, int c) { }
    };
    struct A
    {
      template <class Y>
      static void fooT(Y y) { }

    };

  }
  template <class T>
    struct Bar
    {
      void foo(T t1) { }
      void foo(T t1, T t2) { }
      void foo(int a, int b, int c) { }
      template <class Y>
      void fooT(Y y) { }
    };


  struct B
  {
    template <class Y>
    void barT(Y y) { }
    
  };
  
%}


%template(Bar_d) Bar<double>;
%template(Foo_d) space::Foo<double>;
%template(foo) space::A::fooT<double>;
%template(foo) space::A::fooT<int>;
%template(foo) space::A::fooT<char>;

%template(foo) B::barT<double>;
%template(foo) B::barT<int>;
%template(foo) B::barT<char>;
