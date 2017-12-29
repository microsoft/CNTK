/* File : example.c */

#include "example.h"

/* copy the contents of the first array to the second */
void myArrayCopy( int* sourceArray, int* targetArray, int nitems ) {
  int i;
  for ( i = 0; i < nitems; i++ ) {
    targetArray[ i ] = sourceArray[ i ];
  }
}

/* swap the contents of the two arrays */
void myArraySwap( int* array1, int* array2, int nitems ) {
  int i, temp;
  for ( i = 0; i < nitems; i++ ) {
    temp = array1[ i ];
    array1[ i ] = array2[ i ];
    array2[ i ] = temp;
  }
}

