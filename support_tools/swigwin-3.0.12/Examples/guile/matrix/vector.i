//
// FILE : vector.i

%{
#include "vector.h"
%}

%inline {

extern Vector *createv(double x,double y,double z,double w);
/* Creates a new vector v(x,y,z,w) */

extern void destroyv(Vector *v);
/* Destroys the vector v */

extern void   printv(Vector *v);
/* Prints out the vector v */

extern void   transform(double **T, Vector *v, Vector *t);
/* Transforms vector c to vector t by M*v --> t */

}
