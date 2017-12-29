/* File : example.c */

#include "example.h"
#include <math.h>
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

int Shape::get_nshapes() {
  return nshapes;
}

/* Move the shape to a new location */
void Shape::move(double dx, double dy) {
  x += dx;
  y += dy;
}

int Shape::nshapes = 0;

double Circle::area(void) {
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

const char *overloaded(int i) {
  return "Overloaded with int";
}

const char *overloaded(double d) {
  return "Overloaded with double";
}

const char *overloaded(const char * str) {
  return "Overloaded with char *";
}

const char *overloaded( const Circle& ) {
  return "Overloaded with Circle";
}

const char *overloaded( const Shape& ) {
  return "Overloaded with Shape";
}
