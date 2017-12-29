/*   FILE : matrix.c : some simple 4x4 matrix operations */
#include <stdlib.h>
#include <stdio.h>

double **new_matrix() {

  int i;
  double **M;

  M = (double **) malloc(4*sizeof(double *));
  M[0] = (double *) malloc(16*sizeof(double));
  
  for (i = 0; i < 4; i++) {
    M[i] = M[0] + 4*i;
  }
  return M;
}

void destroy_matrix(double **M) {

  free(M[0]);
  free(M);

}

void print_matrix(double **M) {

  int i,j;

  for (i = 0; i < 4; i++) {
    for (j = 0; j < 4; j++) {
      printf("%10g ", M[i][j]);
    }
    printf("\n");
  }

}

void mat_mult(double **m1, double **m2, double **m3) {

  int i,j,k;
  double temp[4][4];

  for (i = 0; i < 4; i++) 
    for (j = 0; j < 4; j++) {
      temp[i][j] = 0;
      for (k = 0; k < 4; k++) 
	temp[i][j] += m1[i][k]*m2[k][j];
    }

  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++)
      m3[i][j] = temp[i][j];
}







