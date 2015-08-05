#include "stdafx.h"
#include "ColumnQuantizer.h"
#include <cassert>
namespace Microsoft { namespace MSR { namespace CNTK {
void ColumnQuantizerQbwordsPerColTest()
{
	int n32 = ColumnQuantizer::QbwordsPerCol(32, 32);
	int n16 = ColumnQuantizer::QbwordsPerCol(32, 16);
	int n8 = ColumnQuantizer::QbwordsPerCol(32, 8);
	int n4 = ColumnQuantizer::QbwordsPerCol(32, 4);
	int n2 = ColumnQuantizer::QbwordsPerCol(32, 2);
	int n1 = ColumnQuantizer::QbwordsPerCol(32, 1);
	int na1 = ColumnQuantizer::QbwordsPerCol(1, 1);
	int nb1 = ColumnQuantizer::QbwordsPerCol(3, 1);

	assert(n32 == 32);
	assert(n16 == 16);
	assert(n8 == 8);
	assert(n4 == 4);
	assert(n2 == 2);
	assert(n1 == 1);
	assert(na1 == 1);
	assert(nb1 == 1);
}

//test of computing [low, high] range of quantization
void ColumnQuantizerRangeTest()
{
	int n = 10;
	float * matrix = new float[n*n];
	for(int j = 0; j < n; j++)
	{
		for(int i = 0; i < n; i++)
		{
			int ij = j*n + i;
			matrix[ij]= i;
		}
	}
	for(int j = 0; j < n; j++)
	{
		float lower;
		float upper;
		ColumnQuantizer<float>::ComputeRangeStatColj(matrix, n, n, j, 1, lower, upper);
		fprintf(stderr, "%f %f\n",lower, upper);
	}

	for(int j = 0; j < n; j++)
	{
		float lower;
		float upper;
		ColumnQuantizer<float>::ComputeRangeStatColj(matrix, n, n, j, 32, lower, upper);
		fprintf(stderr, "%f %f\n",lower, upper);
	}
}

void ColumnQuantizerQuantizeTest()
{
	int n = 11;
	float * matrix = new float[n*n];
	float * uqMatrix = new float[n*n];
	float * residual = new float[n*n];
	for(int j = 0; j < n; j++)
	{
		for(int i = 0; i < n; i++)
		{
			int ij = j*n + i;
			matrix[ij]= i-n/2;
			residual[ij] = 0.0f;
		}
	}

	int nQB = ColumnQuantizer<float>::QbwordsPerCol(n, 1);
	QBWord *qb = new QBWord[nQB];


	//quantization of 1 bit
	ColumnQuantizer<float> q(0, -10, 10);
	q.Quantize(matrix, residual, n, n, 0, qb, residual);
	q.Unquantize(uqMatrix,n,n,0, qb, false);

	for(int i = 0; i < n; i++)
	{
		int j=0;
		int ij = j*n + i;
		fprintf(stderr, " matrix %f unquan matrix %f residual %f \n", matrix[ij], uqMatrix[ij], residual[ij]);
	}

	//quantization of 32 bit
	memset(residual,0,sizeof(float)*n*n);
	int nQB32 = ColumnQuantizer<float>::QbwordsPerCol(n, 32);
	QBWord *qb32 = new QBWord[nQB32];
	ColumnQuantizer<float> q32(5, -10, 10);
	q32.Quantize(matrix, residual, n, n, 0, qb32, residual);
	q32.Unquantize(uqMatrix,n,n,0, qb32, false);

	for(int i = 0; i < n; i++)
	{
		int j=0;
		int ij = j*n + i;
		fprintf(stderr, " matrix %f unquan matrix %f residual %f \n", matrix[ij], uqMatrix[ij], residual[ij]);
	}
}

void ColumnQuantizerTestMain()
{
	ColumnQuantizerQuantizeTest();
	ColumnQuantizerQbwordsPerColTest();
	ColumnQuantizerRangeTest();
}

/*int _tmain(int argc, _TCHAR* argv[])
{
	ColumnQuantizerTestMain();
}*/
}}}