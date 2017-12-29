// Simple tests of overloaded functions
%module overload_simple

#ifdef SWIGCHICKEN
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) fbool;
#endif

#ifdef SWIGLUA
// lua only has one numeric type, so most of the overloads shadow each other creating warnings
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) foo;
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) bar;
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) Spam;
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) num;
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) fid;
#endif

#ifdef SWIGGO
%warnfilter(SWIGWARN_PARSE_KEYWORD) type; // 'type' is a Go keyword, renamed as 'Xtype'
%rename(Foos) Foo;
#endif

#ifndef SWIG_NO_OVERLOAD
%immutable Spam::type;

%inline %{

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
char *foo(int) {
   return (char*) "foo:int";
}

char *foo(double) {
   return (char*) "foo:double";
}

char *foo(char *) {
   return (char*) "foo:char *";
}

char *foo(Foo *) {
   return (char*) "foo:Foo *";
}
char *foo(Bar *) {
   return (char *) "foo:Bar *";
}
char *foo(void *) {
   return (char *) "foo:void *";
}
char *foo(Foo *, int) {
   return (char *) "foo:Foo *,int";
}
char *foo(double, Bar *) {
   return (char *) "foo:double,Bar *";
}

char *blah(double) {
   return (char *) "blah:double";
}

char *blah(char *) {
   return (char *) "blah:char *";
}

class Spam {
public:
    Spam() { type = "none"; }
    Spam(int) { type = "int"; }
    Spam(double) { type = "double"; }
    Spam(char *) { type = "char *"; }
    Spam(Foo *) { type = "Foo *"; }
    Spam(Bar *) { type = "Bar *"; }
    Spam(void *) { type = "void *"; }
    const char *type;

char *foo(int) {
   return (char*) "foo:int";
}
char *foo(double) {
   return (char*) "foo:double";
}
char *foo(char *) {
   return (char*) "foo:char *";
}
char *foo(Foo *) {
   return (char*) "foo:Foo *";
}
char *foo(Bar *) {
   return (char *) "foo:Bar *";
}
char *foo(void *) {
   return (char *) "foo:void *";
}

static char *bar(int) {
   return (char*) "bar:int";
}
static char *bar(double) {
   return (char*) "bar:double";
}
static char *bar(char *) {
   return (char*) "bar:char *";
}
static char *bar(Foo *) {
   return (char*) "bar:Foo *";
}
static char *bar(Bar *) {
   return (char *) "bar:Bar *";
}
static char *bar(void *) {
   return (char *) "bar:void *";
}
};


bool fbool(bool b) {
   return b;
}

int fbool(int b) {
   return b;
}

char *fint(int) {
   return (char*) "fint:int";
}

char *fdouble(double) {
   return (char*) "fdouble:double";
}

char *num(int) {
   return (char*) "num:int";
}
char *num(double) {
   return (char*) "num:double";
}

char *fid(int, int) {
   return (char*) "fid:intint";
}
char *fid(int, double) {
   return (char*) "fid:intdouble";
}

char *fid(double, int) {
   return (char*) "fid:doubleint";
}

char *fid(double, double) {
   return (char*) "fid:doubledouble";
}

%}

%inline %{
unsigned long long ull() { return 0ULL; }
unsigned long long ull(unsigned long long ull) { return ull; }
long long ll() { return 0LL; }
long long ll(long long ull) { return ull; }
%}

%include cmalloc.i
%malloc(void);
%free(void);

#endif


%inline {  
  class ClassA
  {
  public:
    ClassA() {}  
    int method1( ) {return 0;}
    int method1( int arg1 ) {return arg1;}
  protected:
    int method1( int arg1, int arg2 ) {return arg1 + arg2;}
    
  };
}

#ifdef SWIGPYTHON
%inline 
{
  class Graph {
  public:
    int val;
    Graph(int i) : val(i) {};
  };
}

%extend Graph {
  Graph(PyObject* p) { return new Graph(123);}
} 

#endif

%inline %{
  int int_object(Spam *s) { return 999; }
  int int_object(int c) { return c; }
%}

