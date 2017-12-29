/* File : example.c */

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

int fact(int n) {
  if (n <= 0) return 1;
  return n*fact(n-1);
}
