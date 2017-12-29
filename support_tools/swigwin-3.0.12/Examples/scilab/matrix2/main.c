double sumitems(double *first, int nbRow, int nbCol);
void main(){
/**
 * --> myMatrix=[ 103 3 1    12;0   0 2043 1];
 * --> sumitems(myMatrix);
 * 32
 */
	double B[] = {1,3,4,9,2,8,3,2};   /* Declare the matrix */
	int rowB = 2, colB = 4; 
	printf("sumitems: %6.2f\n",sumitems(B, rowB, colB));


/**
 * --> myOtherMatrix=generateValues();
 * --> size(myOtherMatrix);
 */
	int numberRow, numberCol, i;
	double * matrix=getValues(&numberRow, &numberCol);
	printf("Matrix of size [%d,%d]",numberRow, numberCol);
	for(i=0; i < numberRow*numberCol; i++)
	{
		printf("A[%d] = %5.2f\n",i,matrix[i]);
	}
}
