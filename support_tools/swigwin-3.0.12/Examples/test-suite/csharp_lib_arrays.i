%module csharp_lib_arrays

%include "arrays_csharp.i"

%apply int INPUT[]  { int* sourceArray }
%apply int OUTPUT[] { int* targetArray }

%apply int INOUT[] { int* array1 }
%apply int INOUT[] { int* array2 }

%inline %{
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
%}


%clear int* sourceArray;
%clear int* targetArray;

%clear int* array1;
%clear int* array2;


// Below replicates the above array handling but this time using the pinned (fixed) array typemaps
%csmethodmodifiers myArrayCopyUsingFixedArrays "public unsafe";
%csmethodmodifiers myArraySwapUsingFixedArrays "public unsafe";

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

  
