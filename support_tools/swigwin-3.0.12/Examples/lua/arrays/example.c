/* File : example.c */

#include <stdlib.h>

/* we are using the qsort function, which needs a helper function to sort */
int compare_int(const void * a, const void * b)
{
  return ( *(int*)a - *(int*)b );
}

void sort_int(int* arr, int len)
{
  qsort(arr, len, sizeof(int), compare_int);
}

/* ditto doubles */
int compare_double(const void * a, const void * b)
{
  return (int)( *(double*)a - *(double*)b );
}

void sort_double(double* arr, int len)
{
  qsort(arr, len, sizeof(double), compare_double);
}
