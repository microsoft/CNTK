%module example

%include matrix.i

%apply (double *IN, int IN_ROWCOUNT, int IN_COLCOUNT) { (double *inputMatrix, int nbRow, int nbCol) }
%apply (double **OUT, int *OUT_ROWCOUNT, int *OUT_COLCOUNT) { (double **resultMatrix, int *nbRowRes, int *nbColRes) }

%apply (int *IN, int IN_ROWCOUNT, int IN_COLCOUNT) { (int *inputMatrix, int nbRow, int nbCol) }
%apply (int **OUT, int *OUT_ROWCOUNT, int *OUT_COLCOUNT) { (int **resultMatrix, int *nbRowRes, int *nbColRes) }

%apply (char **IN, int IN_SIZE) { (char **inputVector, int size) }
%apply (char ***OUT, int *OUT_SIZE) { (char ***resultVector, int *sizeRes) }

%inline %{
  extern double sumDoubleMatrix(double *inputMatrix, int nbRow, int nbCol);
  extern void squareDoubleMatrix(double *inputMatrix, int nbRow, int nbCol, double **resultMatrix, int *nbRowRes, int *nbColRes);
  extern void getDoubleMatrix(double **resultMatrix, int *nbRowRes, int *nbColRes);

  extern int sumIntegerMatrix(int *inputMatrix, int nbRow, int nbCol);
  extern void squareIntegerMatrix(int *inputMatrix, int nbRow, int nbCol, int **resultMatrix, int *nbRowRes, int *nbColRes);
  extern void getIntegerMatrix(int **resultMatrix, int *nbRowRes, int *nbColRes);

  extern char* concatStringVector(char **inputVector, int size);
  extern void getStringVector(char ***resultVector, int *sizeRes);
%}

