/* File : example.c */

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

extern "C" void factor( int &x, int &y ) {
    int gcd_xy = gcd( x,y );
    x /= gcd_xy;
    y /= gcd_xy;
}
