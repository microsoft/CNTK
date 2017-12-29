%module d_nativepointers;

%inline %{
  class SomeClass {
  };
  class OpaqueClass;
  typedef void (*FuncA)(int **x, char ***y);
  typedef void (*FuncB)(int **x, SomeClass *y);

  int *a( int *value ){ return value; }
  float **b( float **value ){ return value; }
  char ***c( char ***value ){ return value; }
  SomeClass *d( SomeClass *value ){ return value; }
  SomeClass **e( SomeClass **value ){ return value; }
  OpaqueClass *f( OpaqueClass *value ){ return value; }
  FuncA g( FuncA value ){ return value; }
  FuncB* h( FuncB* value ){ return value; }

  int &refA( int &value ){ return value; }
  float *&refB( float *&value ){ return value; }
%}
