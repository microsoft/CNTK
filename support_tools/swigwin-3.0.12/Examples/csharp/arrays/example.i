/* File : example.i */
%module example

%{
#include "example.h"
%}
%include "arrays_csharp.i"

%apply int INPUT[]  { int* sourceArray }
%apply int OUTPUT[] { int* targetArray }

%apply int INOUT[] { int* array1 }
%apply int INOUT[] { int* array2 }

%include "example.h"

%clear int* sourceArray;
%clear int* targetArray;

%clear int* array1;
%clear int* array2;


// Below replicates the above array handling but this time using the pinned (fixed) array typemaps
%csmethodmodifiers "public unsafe";

%apply int FIXED[] { int* sourceArray }
%apply int FIXED[] { int* targetArray }

%inline %{
void myArrayCopyUsingFixedArrays( int *sourceArray, int* targetArray, int nitems ) {
  myArrayCopy(sourceArray, targetArray, nitems);
}
%}

%apply int FIXED[] { int* array1 }
%apply int FIXED[] { int* array2 }

%inline %{
void myArraySwapUsingFixedArrays( int* array1, int* array2, int nitems ) {
  myArraySwap(array1, array2, nitems);
}
%}

  
