%module exception_partial_info

// This produced compilable code for Tcl, Python in 1.3.27, fails in 1.3.29

%{
#if defined(_MSC_VER)
  #pragma warning(disable: 4290) // C++ exception specification ignored except to indicate a function is not __declspec(nothrow)
#endif
%}

%{
class myException
{
   public:
      virtual const char *name() = 0;
};

class ex1 : public myException
{
   public:
      virtual const char *name() { return "ex1"; }
};

class ex2 : public myException
{
   public:
      virtual const char *name()  { return "ex2"; }
};
%}

#if !defined(SWIGUTL)

#if !defined(SWIGCHICKEN)

%inline %{
class Impl
{
   public:
      void f1() throw (myException) { ex1 e; throw e; }
      void f2() throw (myException) { ex2 e; throw e; }
};
%}

#else
#warning "Chicken needs fixing for partial exception information"
#endif

#else
#warning "UTL needs fixing for partial exception information"
#endif

