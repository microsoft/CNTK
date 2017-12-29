/* File : example.h */

extern void foo (int x);
extern void foo (char *x);

class Foo {
 private:
  int myvar;
 public:
  Foo();
  Foo(const Foo &);   // Copy constructor
  void bar(int x);
  void bar(char *s, int y);
};
