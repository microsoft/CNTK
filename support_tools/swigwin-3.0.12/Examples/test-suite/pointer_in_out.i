/* This file tests the pointer-in-out typemap library,
   currently only available for Guile. */

%module pointer_in_out

%include "pointer-in-out.i"

TYPEMAP_POINTER_INPUT_OUTPUT(int *, int-pointer);

int consume_int_pointer(int **INPUT);
void produce_int_pointer(int **OUTPUT, int value1, int value2);
void frobnicate_int_pointer(int **INOUT);

%{

int consume_int_pointer(int **INPUT)
{
  return **INPUT;
}

void produce_int_pointer(int **OUTPUT, int value1, int value2)
{
  int *foo = malloc(2 * sizeof(int));
  foo[0] = value1;
  foo[1] = value2;
  *OUTPUT = foo;
}

void frobnicate_int_pointer(int **INOUT)
{
  /* advance the pointer */
  (*INOUT)++;
}

%}
