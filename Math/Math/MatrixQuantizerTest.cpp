#include "stdafx.h"
#include "MatrixQuantizer.h"
#include <chrono>
#include <iostream>

using namespace Microsoft::MSR::CNTK;
void MatrixQuantizerQuantizeTest(int nBits, short deviceID = CPUDEVICE, int nRow=11, int nCol=11)
{
	using namespace std::chrono;
	//initialize the matrix
	float * matrix = new float[nRow*nCol];
	float * uqMatrix = new float[nRow*nCol];
	float * residual = new float[nRow*nCol];
	for(int j = 0; j < nCol; j++)
	{
		for (int i = 0; i < nRow; i++)
		{
			int ij = j*nRow + i;
			matrix[ij]= i-nCol/2;
			residual[ij] = 0.0f;
		}
	}
	Matrix<float> m(nRow, nCol, deviceID, MatrixType::DENSE);
	m.SetValueFromCPU(nRow, nCol, matrix);

	
	//residual 
	Matrix<float> r(nRow, nCol, deviceID, MatrixType::DENSE);
	Matrix<float> o(nRow, nCol, deviceID, MatrixType::DENSE);


	//quantized matrix
	size_t qcolSize = QuantizedColumn<ElemType>::QuantizedColumnSize(nBits, nRow);
	QuantizedMatrixCPU qMat(nRow, nCol, qcolSize);


	//quantization
	MatrixQuantizer<float> mq(nRow, nCol, nBits, deviceID);
	Timer t;
	mq.QuantizeAndFetchAsync(m, qMat);
	mq.WaitQuantizeAndFetchAsyncDone();
	double tquant = t.ElapseFromLastCall();
	
	//unquantize
	mq.AssignQuantizedMatrixAsync(qMat);
	mq.UnquantizeAsync(o,false);
	mq.WaitUnquantizeAsyncDone();
	double tunquant = t.ElapseFromLastCall();
	std::cout << "device " << (deviceID<0 ? "CPU" : "GPU") << "quant:" << tquant << " unquant" << tunquant << std::endl;



	if (nCol < 15 && nRow < 15)
	{
		//
		float * uqmatrix = new float[nRow, nCol];
		float *oPtr = deviceID >= 0 ? uqMatrix : o.GetArray();

		if (deviceID >= 0)
		{
			cudaMemcpy(oPtr, o.GetArray(), o.GetNumElements()*sizeof(float), cudaMemcpyDeviceToHost);
		}


		fprintf(stderr, "============================\n");
		for (int j = 0; j < nCol; j++)
		{
			for (int i = 0; i < nRow; i++)
			{
				int ij = j*nRow + i;
				fprintf(stderr, "[%.1f=>%.1f] ", matrix[ij], oPtr[ij]);
			}
			fprintf(stderr, "\n");
		}
	}
}


//test if cpu and gpu produce the same output
void MatrixQuantizerQuantizeTestSame(int nBits, short deviceID = CPUDEVICE, int nRow = 11, int nCol = 11)
{
	//initialize the matrix
	float * matrix = new float[nRow*nCol];
	float * uqMatrix = new float[nRow*nCol];
	float * residual = new float[nRow*nCol];
	for (int j = 0; j < nCol; j++)
	{
		for (int i = 0; i < nRow; i++)
		{
			int ij = j*nRow + i;
			matrix[ij] = i - nCol / 2;
			residual[ij] = 0.0f;
		}
	}
	Matrix<float> m(nRow, nCol, deviceID, MatrixType::DENSE);
	m.SetValueFromCPU(nRow, nCol, matrix);


	Matrix<float> mcpu(nRow, nCol, CPUDEVICE, MatrixType::DENSE);
	mcpu.SetValueFromCPU(nRow, nCol, matrix);


	//residual 
	Matrix<float> r(nRow, nCol, deviceID, MatrixType::DENSE);
	Matrix<float> o(nRow, nCol, deviceID, MatrixType::DENSE);
	Matrix<float> rcpu(nRow, nCol, CPUDEVICE, MatrixType::DENSE);
	Matrix<float> ocpu(nRow, nCol, CPUDEVICE, MatrixType::DENSE);


	//quantized matrix
	size_t qcolSize = QuantizedColumn<ElemType>::QuantizedColumnSize(nBits, nRow);
	QuantizedMatrixCPU qMat(nRow, nCol, qcolSize);
	QuantizedMatrixCPU qMat2(nRow, nCol, qcolSize);


	//quantization
	MatrixQuantizer<float> mq(nRow, nCol, nBits, deviceID);
	mq.QuantizeAndFetchAsync(m, qMat);
	mq.WaitQuantizeAndFetchAsyncDone();

	MatrixQuantizer<float> mqcpu(nRow, nCol, nBits, CPUDEVICE);
	mqcpu.QuantizeAndFetchAsync(mcpu, qMat2);
	mqcpu.WaitQuantizeAndFetchAsyncDone();
	
	//checking now
	int xx = memcmp(qMat.data(), qMat2.data(), qMat.size());


	//unquantize
	mq.AssignQuantizedMatrixAsync(qMat);
	mq.UnquantizeAsync(o, false);
	mq.WaitUnquantizeAsyncDone();

	mqcpu.AssignQuantizedMatrixAsync(qMat2);
	mqcpu.UnquantizeAsync(ocpu, false);
	mqcpu.WaitUnquantizeAsyncDone();

	//
	float * uqmatrix = new float[nRow, nCol];
	float *oPtr = deviceID >= 0 ? uqMatrix : o.GetArray();

	if (deviceID >= 0)
	{
		cudaMemcpy(oPtr, o.GetArray(), o.GetNumElements()*sizeof(float), cudaMemcpyDeviceToHost);
	}


	fprintf(stderr, "============================\n");
	
	for (int j = 0; j < nCol; j++)
	{
		bool mismatch = false;
		for (int i = 0; i < nRow; i++)
		{
			int ij = j*nRow + i;
			//fprintf(stderr, "[%.1f=>%.1f %.1f] ", matrix[ij], oPtr[ij], ocpu.GetArray()[ij]);
			if (fabsf(oPtr[ij] - ocpu.GetArray()[ij]) > 1e-3)
			{
				fprintf(stderr, "[%.1f=>%.1f %.1f] ", matrix[ij], oPtr[ij], ocpu.GetArray()[ij]);
				mismatch = true;
			}
		}
		if (mismatch) fprintf(stderr, "\n");
	}
}
 
void MatrixQuantizerTestMain()
{
	//MatrixQuantizerQuantizeTestSame(1, 1,100,200);
	MatrixQuantizerQuantizeTest(1, 1,2000,2000);
	//MatrixQuantizerQuantizeTest(1,CPUDEVICE,2000,2000);
	//MatrixQuantizerQuantizeTest(1);
	//MatrixQuantizerQuantizeTest(32);
}

int xmain()
{
	MatrixQuantizerTestMain();
	return 0;
}