/* File : vector.c */

#include <stdlib.h>
#include <stdio.h>
#include "vector.h"

Vector *createv(double x, double y, double z, double w) {

  Vector *n;
  n = (Vector *) malloc(sizeof(Vector));
  n->x = x;
  n->y = y;
  n->z = z;
  n->w = w;
  return n;

}

/* Destroy vector */

void destroyv(Vector *v) {
  free(v);
}

/* Print a vector */

void printv(Vector *v) {

  printf("x = %g, y = %g, z = %g, w = %g\n", v->x, v->y, v->z, v->w);

}

/* Do a transformation */
void transform(double **m, Vector *v, Vector *r) {

  r->x = m[0][0]*v->x + m[0][1]*v->y + m[0][2]*v->z + m[0][3]*v->w;
  r->y = m[1][0]*v->x + m[1][1]*v->y + m[1][2]*v->z + m[1][3]*v->w;
  r->z = m[2][0]*v->x + m[2][1]*v->y + m[2][2]*v->z + m[2][3]*v->w;
  r->w = m[3][0]*v->x + m[3][1]*v->y + m[3][2]*v->z + m[3][3]*v->w;

}



