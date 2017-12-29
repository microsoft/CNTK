%module callback

// Not specifying the callback name is only possible in Python.
#ifdef SWIGPYTHON
%callback(1) foo;
%callback(1) foof;
%callback(1) A::bar;
%callback(1) A::foom;
#else
%callback("%s") foo;
%callback("%s") foof;
%callback("%s") A::bar;
%callback("%s") A::foom;
#endif
%callback("%(uppercase)s_Cb_Ptr") foo_T;  // this works in Python too

%inline %{

  int foo(int a) {
    return a;
  }
  
  int foof(int a) {
    return 3*a;
  }  
  
  struct A 
  {
    static int bar(int a) {
      return 2*a;
    }

    int foom(int a) 
    {
      return -a;
    }

    //friend int foof(int a);
  };


  extern "C" int foobar(int a, int (*pf)(int a)) {
    return pf(a);
  }

  extern "C" int foobarm(int a, A ap, int (A::*pf)(int a)) {
    return (ap.*pf)(a);
  }

  template <class T>
    T foo_T(T a) 
    {
      return a;
    }

  template <class T>
    T foo_T(T a, T b) 
    {
      return a + b;
    }


  template <class T>
  T foobar_T(T a, T (*pf)(T a)) {
    return pf(a);
  }

#if defined(__SUNPRO_CC)
// workaround for: Error: Could not find a match for foobar_T<T>(int, extern "C" int(*)(int)).
  extern "C" {
    typedef int (*foobar_int_int)(int a);
    typedef double (*foobar_double_double)(double a);
  };
  template <class T>
  int foobar_T(int a, foobar_int_int pf) {
    return pf(a);
  }
  template <class T>
  double foobar_T(double a, foobar_double_double pf) {
    return pf(a);
  }
#endif

  template <class T>
  const T& ident(const T& x) {
    return x;
  }
%}

%template(foo_i) foo_T<int>;
%template(foobar_i) foobar_T<int>;

%template(foo_d) foo_T<double>;
%template(foobar_d) foobar_T<double>;

%template(ident_d) ident<double>;
