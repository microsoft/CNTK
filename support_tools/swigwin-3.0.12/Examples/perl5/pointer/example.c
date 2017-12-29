/* File : example.c */

void add(int *x, int *y, int *result) {
  *result = *x + *y;
}

void subtract(int *x, int *y, int *result) {
  *result = *x - *y;
}

int divide(int n, int d, int *r) {
   int q;
   q = n/d;
   *r = n - q*d;
   return q;
}
