%module scilab_li_matrix

%include "matrix.i"

%define %use_matrix_apply(TYPE...)
%apply (TYPE *IN, int IN_ROWCOUNT, int IN_COLCOUNT) { (TYPE *matrix, int nbRow, int nbCol) }
%apply (TYPE **OUT, int *OUT_ROWCOUNT, int *OUT_COLCOUNT) { (TYPE **matrixRes, int *nbRowRes, int *nbColRes) }
%apply (TYPE *IN, int IN_SIZE) { (TYPE *matrix, int size) }
%apply (TYPE **OUT, int *OUT_SIZE) { (TYPE **matrixRes, int *sizeRes) }
%enddef

%use_matrix_apply(int);
%use_matrix_apply(double);
%use_matrix_apply(char *);
%use_matrix_apply(bool);

%inline %{

/*
 * (T *matrixIn, int matrixInSize) pattern functions
 */

template<typename T> T inMatrixSize(T *matrix, int size) {
  T sum = 0;
  int i;
  for (i = 0; i < size; i++)
    sum += matrix[i];
  return sum;
}

template<typename T> void outMatrixSize(T **matrixRes, int *sizeRes) {
  int size;
  int i;
  *sizeRes = 6;
    *matrixRes = (T*) malloc(*sizeRes * sizeof(T));
  for (i = 0; i < *sizeRes; i++)
    (*matrixRes)[i] = i;
}

template<typename T> void inoutMatrixSize(T *matrix, int size, T **matrixRes, int *sizeRes) {
  int i;
  *sizeRes = size;
  *matrixRes = (T*) malloc(size * sizeof(T));
  for (i = 0; i < size; i++) {
    (*matrixRes)[i] = matrix[i] * matrix[i];
  }
}

/*
 * (char **matrixIn, int matrixInSize) pattern functions
 */

template<> char* inMatrixSize(char **matrix, int size) {
  char *s = (char *) calloc(size + 1, sizeof(char));
  int i;
  for (i = 0; i < size; i++)
    strcat(s, matrix[i]);
  return s;
}

template<> void outMatrixSize(char ***matrixRes, int *sizeRes) {
  char *s;
  int i;
  *sizeRes = 6;
  *matrixRes = (char **) malloc(*sizeRes * sizeof(char *));
  for (i = 0; i < *sizeRes; i++) {
    s = (char *) malloc(sizeof(char)+1);
    sprintf(s, "%c", char(65 + i));
    (*matrixRes)[i] = s;
  }
}

template<> void inoutMatrixSize(char **matrix, int size,
  char ***matrixRes, int *sizeRes) {
  int i;
  char *s;
  *sizeRes = size;
  *matrixRes = (char **) malloc(*sizeRes * sizeof(char* ));
  for (i = 0; i < *sizeRes; i++) {
    s = (char *) malloc((2 * strlen(matrix[i]) + 1) * sizeof(char));
    sprintf(s, "%s%s", matrix[i], matrix[i]);
    (*matrixRes)[i] = s;
  }
}

/*
 * (bool **matrixIn, int matrixInSize) pattern functions
 */

template<> bool inMatrixSize(bool *matrix, int size) {
  bool b = true;
  int i;
  b = matrix[0];
  for (i = 1; i < size; i++)
    b = b ^ matrix[i];
  return b;
}

template<> void outMatrixSize(bool **matrixRes, int *sizeRes) {
  int i;
  *sizeRes = 6;
  *matrixRes = (bool*) malloc(*sizeRes * sizeof(bool));
  for (i = 0; i < *sizeRes; i++) {
    (*matrixRes)[i] = i % 2 == 0;
  }
}

template<> void inoutMatrixSize(bool *matrix, int size,
  bool **matrixRes, int *sizeRes) {
  int i;
  *sizeRes = size;
  *matrixRes = (bool*) malloc(*sizeRes * sizeof(bool));
  for (i = 0; i < *sizeRes; i++) {
    (*matrixRes)[i] = matrix[i] == 1 ? 0 : 1;
  }
}

/*
 * (T *matrixIn, int matrixInRowCount, int matrixInColCount) pattern functions
 */

template<typename T> T inMatrixDims(T *matrix, int nbRow, int nbCol) {
  return inMatrixSize(matrix, nbRow * nbCol);
}

template<typename T> void outMatrixDims(T **matrixRes, int *nbRowRes, int *nbColRes) {
  int size = 0;
  outMatrixSize(matrixRes, &size);
  *nbRowRes = 3;
  *nbColRes = 2;
}

template<typename T> void inoutMatrixDims(T *matrix, int nbRow, int nbCol,
  T **matrixRes, int *nbRowRes, int *nbColRes) {
  *nbRowRes = nbRow;
  *nbColRes = nbCol;
  int sizeRes = 0;
  inoutMatrixSize(matrix, nbRow * nbCol, matrixRes, &sizeRes);
}

%}

%define %instantiate_matrix_template_functions(NAME, TYPE...)
%template(in ## NAME ## MatrixDims) inMatrixDims<TYPE>;
%template(out ## NAME ## MatrixDims) outMatrixDims<TYPE>;
%template(inout ## NAME ## MatrixDims) inoutMatrixDims<TYPE>;
%template(in ## NAME ## MatrixSize) inMatrixSize<TYPE>;
%template(out ## NAME ## MatrixSize) outMatrixSize<TYPE>;
%template(inout ## NAME ## MatrixSize) inoutMatrixSize<TYPE>;
%enddef

%instantiate_matrix_template_functions(Int, int);
%instantiate_matrix_template_functions(Double, double);
%instantiate_matrix_template_functions(CharPtr, char *);
%instantiate_matrix_template_functions(Bool, bool);






