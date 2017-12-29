/* File : example.c */

#include "example.h"

double dot_product(Vector a, Vector b) {
  return (a.x*b.x + a.y*b.y + a.z*b.z);
}

void vector_add(Vector a, Vector b, Vector* result) {
  result->x = a.x + b.x;
  result->y = a.y + b.y;
  result->z = a.z + b.z;
}
