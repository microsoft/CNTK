%module xxx

int foo(int x, int y);
int foo;

int bar(int x);

struct bar {
  int y;
};

%rename(bar) spam;

int spam(int);






