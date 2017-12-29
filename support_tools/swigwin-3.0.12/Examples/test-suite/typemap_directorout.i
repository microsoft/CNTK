// Test named output typemaps used in directors are found during the typematch search
%module(directors="1") typemap_directorout

%warnfilter(SWIGWARN_TYPEMAP_THREAD_UNSAFE,SWIGWARN_TYPEMAP_DIRECTOROUT_PTR) Class1;

%feature("director");

%typemap(out) MyType & %{ WILL_NOT_COMPILE %}
%typemap(out) MyType &USEME = SWIGTYPE &;
%typemap(out) MyType &Class1::foo2, MyType &foo1 %{ /* special start */ $typemap(out, MyType &USEME) /* special end */ %}

%typemap(directorout) MyType & %{ WILL_NOT_COMPILE %}
// Can't use the %typemap(directorout) MyType & = SWIGTYPE & approach as non-director languages don't define any directorout typemaps
%typemap(directorout) MyType &Class1::foo2, MyType &foo1 %{ /* special start */ $input = 0; /* special end */ %}


#ifdef SWIGCSHARP
%typemap(csdirectorout) MyType & %{ WILL_NOT_COMPILE %}
%typemap(csdirectorout) MyType &USEME = SWIGTYPE &;
%typemap(csdirectorout) MyType &Class1::foo2, MyType &foo1 %{ /* special start */ $typemap(csdirectorout, MyType &USEME) /* special end */ %}
#endif

#ifdef SWIGD
%typemap(ddirectorout) MyType & %{ WILL_NOT_COMPILE %}
%typemap(ddirectorout) MyType &USEME = SWIGTYPE &;
%typemap(ddirectorout, nativepointer="cast(void*)$dcall") MyType &Class1::foo2, MyType &foo1 %{ /* special start */ $typemap(ddirectorout, MyType &USEME) /* special end */ %}
#endif

#ifdef SWIGJAVA
%typemap(javadirectorout) MyType & %{ WILL_NOT_COMPILE %}
%typemap(javadirectorout) MyType &USEME = SWIGTYPE &;
%typemap(javadirectorout) MyType &Class1::foo2, MyType &foo1 %{ /* special start */ $typemap(javadirectorout, MyType &USEME) /* special end */ %}
#endif

%inline %{
typedef int MyType;
class Class1
{
  MyType mt;
public:
  Class1() : mt() {}
  virtual MyType & foo1() { return mt; }
  virtual MyType & foo2(int parm1) { return mt; }
  virtual MyType & foo2() { return mt; }
  virtual ~Class1() {}
};
%}

