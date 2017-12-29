/* File : example.c */

#include <stdio.h>

/* A global variable */
double Foo = 3.0;

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

void greeting() {
  printf("Hello from the C function 'greeting'\n");
}
