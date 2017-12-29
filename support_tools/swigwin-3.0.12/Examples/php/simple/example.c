/* File : example.c */
#include <stdio.h>

/* A global variable */
double Foo = 3.0;

void print_Foo() {
   printf("In C, Foo = %f\n",Foo);
}

/* Compute the greatest common divisor of positive integers */
int gcd(int x, int y) {
  int g;
  g = y;
  while (x > 0) {
    g = x;
    x = y % x;
    y = g;
  }
  return g;
}


