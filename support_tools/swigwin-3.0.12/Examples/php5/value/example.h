/* File : example.h */

typedef struct {
     double x, y, z;
} Vector;

double dot_product(Vector a, Vector b);
void vector_add(Vector a, Vector b, Vector* result);
