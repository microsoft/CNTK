// Tests SWIG's handling of pass-by-value for complex datatypes
%module example

%{
#include "example.h"
%}

/* Some functions that manipulate Vectors by value */
%inline %{
extern double dot_product(Vector a, Vector b);
extern Vector vector_add(Vector a, Vector b);
%}

/* Include this because the vector_add() function will leak memory */
void   free(void *);

/* Some helper functions for our interface */
%inline %{

Vector *new_Vector(double x, double y, double z) {
   Vector *v = (Vector *) malloc(sizeof(Vector));
   v->x = x;
   v->y = y;
   v->z = z;
   return v;
}

void vector_print(Vector *v) {
  printf("Vector %p = (%g, %g, %g)\n", (void *)v, v->x, v->y, v->z);
}
%}

