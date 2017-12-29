// Tests of overloaded functions of arrays
// Based on overload_simple testcase
%module overload_arrays

#ifdef SWIGCHICKEN
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) fbool;
#endif

#ifdef SWIGLUA
// lua only has one numeric type, so most of the overloads shadow each other creating warnings
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) foo;
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) bar;
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) Spam;
#endif

#ifdef SWIGGO
%warnfilter(SWIGWARN_PARSE_KEYWORD) type; // 'type' is a Go keyword, renamed as 'Xtype'
%rename(Foos) Foo;
#endif



#ifndef SWIG_NO_OVERLOAD
%immutable Spam::type;

%inline %{

#define SIZE 3

struct Foo {
};

class Bar {
public:
  Bar(int i = 0) { num = i; }

  static int foo(int a=0, int b=0) {return 0;}

  int num;
};

char *foo() {
   return (char *) "foo:";
}
char *foo(int[SIZE]) {
   return (char*) "foo:int[SIZE]";
}

char *foo(double[SIZE]) {
   return (char*) "foo:double[SIZE]";
}

char *foo(char *[SIZE]) {
   return (char*) "foo:char *[SIZE]";
}

char *foo(Foo *[SIZE]) {
   return (char*) "foo:Foo *[SIZE]";
}
char *foo(Bar *[SIZE]) {
   return (char *) "foo:Bar *[SIZE]";
}
char *foo(void *[SIZE]) {
   return (char *) "foo:void *[SIZE]";
}
char *foo(Foo *[SIZE], int[SIZE]) {
   return (char *) "foo:Foo *[SIZE],int[SIZE]";
}
char *foo(double[SIZE], Bar *[SIZE]) {
   return (char *) "foo:double[SIZE],Bar *[SIZE]";
}

char *blah(double[SIZE]) {
   return (char *) "blah:double[SIZE]";
}

char *blah(char *[SIZE]) {
   return (char *) "blah:char *[SIZE]";
}

class Spam {
public:
    Spam() { type = "none"; }
    Spam(int[SIZE]) { type = "int[SIZE]"; }
    Spam(double[SIZE]) { type = "double[SIZE]"; }
    Spam(char *[SIZE]) { type = "char *[SIZE]"; }
    Spam(Foo *[SIZE]) { type = "Foo *[SIZE]"; }
    Spam(Bar *[SIZE]) { type = "Bar *[SIZE]"; }
    Spam(void *[SIZE]) { type = "void *[SIZE]"; }
    const char *type;

char *foo(int[SIZE]) {
   return (char*) "foo:int[SIZE]";
}
char *foo(double[SIZE]) {
   return (char*) "foo:double[SIZE]";
}
char *foo(char *[SIZE]) {
   return (char*) "foo:char *[SIZE]";
}
char *foo(Foo *[SIZE]) {
   return (char*) "foo:Foo *[SIZE]";
}
char *foo(Bar *[SIZE]) {
   return (char *) "foo:Bar *[SIZE]";
}
char *foo(void *[SIZE]) {
   return (char *) "foo:void *[SIZE]";
}

static char *bar(int[SIZE]) {
   return (char*) "bar:int[SIZE]";
}
static char *bar(double[SIZE]) {
   return (char*) "bar:double[SIZE]";
}
static char *bar(char *[SIZE]) {
   return (char*) "bar:char *[SIZE]";
}
static char *bar(Foo *[SIZE]) {
   return (char*) "bar:Foo *[SIZE]";
}
static char *bar(Bar *[SIZE]) {
   return (char *) "bar:Bar *[SIZE]";
}
static char *bar(void *[SIZE]) {
   return (char *) "bar:void *[SIZE]";
}
};

%}

#endif


%inline {
  class ClassA
  {
  public:
    ClassA() {}
    int method1( ) {return 0;}
    int method1( int arg1[SIZE] ) {return arg1[0];}
  protected:
    int method1( int arg1[SIZE], int arg2[SIZE] ) {return arg1[0] + arg2[0];}

  };
}

