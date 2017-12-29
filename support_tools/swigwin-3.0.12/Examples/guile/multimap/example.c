/* File : example.c */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

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

int gcdmain(int argc, char *argv[]) {
  int x,y;
  if (argc != 3) {
    printf("usage: gcd x y\n");
    return -1;
  }
  x = atoi(argv[1]);
  y = atoi(argv[2]);
  printf("gcd(%d,%d) = %d\n", x,y,gcd(x,y));
  return 0;
}

int count(char *bytes, int len, char c) {
  int i;
  int count = 0;
  for (i = 0; i < len; i++) {
    if (bytes[i] == c) count++;
  }
  return count;
}

void capitalize(char *str, int len) {
  int i;
  for (i = 0; i < len; i++) {
    str[i] = (char)toupper(str[i]);
  }
}

void circle(double x, double y) {
  double a = x*x + y*y;
  if (a > 1.0) {
    printf("Bad points %g, %g\n", x,y);
  } else {
    printf("Good points %g, %g\n", x,y);
  }
}
