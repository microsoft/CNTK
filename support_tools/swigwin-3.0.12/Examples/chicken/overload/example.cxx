/* File : example.c */

#include "example.h"
#include <stdio.h>

void foo(int x) {
  printf("x is %d\n", x);
}

void foo(char *x) {
  printf("x is '%s'\n", x);
}

Foo::Foo () {
  myvar = 55;
  printf ("Foo constructor called\n");
}

Foo::Foo (const Foo &) {
  myvar = 66;
  printf ("Foo copy constructor called\n");
}

void Foo::bar (int x) {
  printf ("Foo::bar(x) method ... \n");
  printf("x is %d\n", x);
}

void Foo::bar (char *s, int y) {
  printf ("Foo::bar(s,y) method ... \n");
  printf ("s is '%s'\n", s);
  printf ("y is %d\n", y);
}
