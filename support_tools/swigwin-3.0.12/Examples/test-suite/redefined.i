%module redefined

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) agua;

/* no redundant warnings */
%warnfilter(SWIGWARN_PARSE_REDUNDANT);

#if 1
 //
 // All these repeated declarations are not redefinitions,
 // and they are valid C++ code, therefore, we skip 
 // swig redefined warnings.
 //
%define uja
  aju;
%enddef

%define uja
  aju;
%enddef

%constant int agua = 0;
%constant int agua = 0;

%inline %{

#define REDUNDANT 1
#define REDUNDANT 1

#define MACROREP(x) x
#define MACROREP(x) x

  class Hello;
  class Hello;
  
  typedef int Int;
  typedef int Int;

  inline int hello(int);
  int hello(int) { return 0; }
  
  struct B;
  
  struct A
  {
    typedef int Int;
    friend int foo(A*, B*);    
  };

  struct B
  {
    typedef int Int;
    friend int foo(A*, B*);
  };

  inline int foo(A*, B*) { return 0; }
  
%}


#else

//
// the %extend and %rename directive ALWAYS emit redefined warnings,
// since they are not C/C++/CPP standard.
//
%extend Hello {
  int hi(int) { return 0; }
}

%rename(chao) hi(int);

//
// All these repeated declarations are really redefinitions,
// therefore, swig must produce a redefined warning
//

%constant int agua = 0;
%constant int agua = 1;


%inline %{

#define REDEFINED 1
#define REDEFINED 2

#define MACROREP(x) x
#define MACROREP(x) x*2

  typedef int Int;
  typedef double Int;

  int hi(int);
  int chao(int);
  int hello(int);
  inline double hello(int) { return 0; }
  
  struct Hello 
  {
    typedef int Int;
    typedef double Int;
    friend short hello(int);
    int hi(int) { return 0; }
  };
  
%}
#endif
