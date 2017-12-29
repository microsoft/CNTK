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

void Circle::set_radius( double r ) {
  radius = r;
}

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

ShapeContainer::~ShapeContainer() {
  iterator i=shapes.begin();
  for( iterator i = shapes.begin(); i != shapes.end(); ++i ) {
    delete *i;
  }
}

void
ShapeContainer::addShape( Shape *s ) {
  shapes.push_back( s );
}
