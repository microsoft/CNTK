/* File : example.c */

#include "example.h"

double dot_product(Vector a, Vector b) {
  return (a.x*b.x + a.y*b.y + a.z*b.z);
}

Vector vector_add(Vector a, Vector b) {
  Vector r;
  r.x = a.x + b.x;
  r.y = a.y + b.y;
  r.z = a.z + b.z;
  return r;
}
