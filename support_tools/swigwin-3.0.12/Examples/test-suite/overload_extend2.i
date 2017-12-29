%module overload_extend2

%typemap(default) int int2 "$1=1000;";

%inline %{
typedef struct Foo {
  int dummy;
} Foo;
%}

%extend Foo {
    int test(int x) { x = 0; return 1; }
    int test(char *s) { s = 0; return 2; }
    int test(double x, double y) { x = 0; y = 0; return 3; }
    int test(char *s, int int1, int int2) { s = 0; return int1+int2; }

    /* C default arguments */
    int test(Foo* f, int i=10, int j=20) { return i+j; }
};


