#include "example.h"

#define M_PI 3.14159265358979323846

/* Move the shape to a new location */
void Shape::move(double dx, double dy) {
  x += dx;
  y += dy;
}

int Shape::nshapes = 0;

double Circle::area(void) {
  /* return -1 is to test post-assertion */
  if (radius == 1)
    return -1;
  return M_PI*radius*radius;
}

double Circle::perimeter(void) {
  return 2*M_PI*radius;
}

double Square::area(void) {
  return width*width;
}

double Square::perimeter(void) {
  return 4*width;
}
