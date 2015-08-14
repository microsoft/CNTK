//
// <copyright file="CPUMatrixUnitTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#include "stdafx.h"
#include "CppUnitTest.h"

#include "..\Math\MatrixQuantizer.h"
#include "..\Math\CUDAPageLockedMemAllocator.h"

#define DEBUG_FLAG 1
using namespace Microsoft::MSR::CNTK;

#pragma warning (disable: 4305)

using namespace Microsoft::MSR::CNTK;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNTKMathTest
{
    TEST_CLASS(MatrixQuantizerTests)
    {
    private:

        static const float SINGLE_PRECISION_TOLERANCE;
        static const double DOUBLE_PRECISION_TOLERANCE;

        template <typename ElemType>
        static void ReferenceCPU1BitQuantizer(
            size_t numRows,
            size_t numCols,
            const ElemType* inMatrix,
            const ElemType* prevResidualMatrix,
            const ElemType* prevOutMatrix,
            ElemType* outMatrix,
            ElemType* outResidualMatrix)
        {
            for (size_t j = 0; j < numCols; j++)
            {
                ElemType mean = 0.0f;

                // Calculate the mean0 and mean1 for each column
                ElemType mean0Sum = 0.0f;
                ElemType mean1Sum = 0.0f;
                int num0 = 0;
                int num1 = 0;
                for (int i = 0; i < numRows; i++)
                {
                    size_t flatIdx = (j * numRows) + i;
                    ElemType val = inMatrix[flatIdx];
                    if (val < mean)
                    {
                        mean0Sum += val;
                        num0++;
                    }
                    else
                    {
                        mean1Sum += val;
                        num1++;
                    }
                }

                if (num0 == 0) num0 = 1;                        // happens for all-zero columns which do exist (mean0 is 0 in that case)
                if (num1 == 0) num1 = 1;
                const ElemType mean0 = mean0Sum / num0;
                const ElemType mean1 = mean1Sum / num1;

                for (int i = 0; i < numRows; i++)
                {
                    size_t flatIdx = (j * numRows) + i;
                    ElemType val = inMatrix[flatIdx] + prevResidualMatrix[flatIdx];
                    ElemType qVal;
                    if (val < mean)
                    {
                        qVal = mean0;
                    }
                    else
                    {
                        qVal = mean1;
                    }

                    outMatrix[flatIdx] = prevOutMatrix[flatIdx] + qVal;
                    outResidualMatrix[flatIdx] = val - qVal;
                }
            }
        }

        template <typename ElemType>
        static void Test1BitQuantization(
            size_t numRows,
            size_t numCols,
            ElemType rangeLow,
            ElemType rangeHigh,
            int seed,
            int numIterations,
            short deviceId)
        {
            auto verifyAllZerosFunc = [](Matrix<ElemType>& matrix) {
                ElemType* cpuMatrix = matrix.CopyToArray();
                size_t numMatrixElems = matrix.GetNumElements();
                for (size_t i = 0; i < numMatrixElems; ++i)
                {
                    Assert::IsTrue(cpuMatrix[i] == ((ElemType)0));
                }

                delete[] cpuMatrix;
            };

            MemAllocator* allocator = nullptr;
            if (deviceId != CPUDEVICE)
            {
                allocator = new CUDAPageLockedMemAllocator(deviceId);
            }

            Matrix<ElemType> inMatrix(numRows, numCols, deviceId);
            auto quantizer = MatrixQuantizer<ElemType>::CreateMatrixQuantizer(inMatrix);

            // Verify that the initial residue is comprised of all zeros
            verifyAllZerosFunc(quantizer->GetResidualMatrix());

            Matrix<ElemType> outMatrix(numRows, numCols, deviceId);
            // Verify that the outMatrix is initialized with all zeros
            verifyAllZerosFunc(outMatrix);
            for (int iterNum = 0; iterNum < numIterations; ++iterNum)
            {
                inMatrix = Matrix<ElemType>::RandomUniform(numRows, numCols, rangeLow, rangeHigh, seed + iterNum, deviceId);

                ElemType* gpuInMatrix = inMatrix.CopyToArray();
                ElemType* gpuPrevResidualMatrix = quantizer->GetResidualMatrix().CopyToArray();
                ElemType *gpuPrevOutMatrix = outMatrix.CopyToArray();

                QuantizedMatrix<ElemType> tempCPUQuantizationBuffer(numRows, numCols, 1, CPUDEVICE, allocator);

                quantizer->QuantizeAsync(tempCPUQuantizationBuffer);
                quantizer->WaitQuantizeAsyncDone();

                quantizer->UnquantizeAsync(tempCPUQuantizationBuffer, outMatrix, (iterNum > 0));
                quantizer->WaitUnquantizeAsyncDone();

                // Now verify the quantization results
                ElemType* gpuNewResidualMatrix = quantizer->GetResidualMatrix().CopyToArray();
                ElemType* gpuNewOutMatrix = outMatrix.CopyToArray();

                ElemType PRECISION_TOLERANCE = (sizeof(ElemType) == sizeof(double)) ? ((ElemType)DOUBLE_PRECISION_TOLERANCE) : SINGLE_PRECISION_TOLERANCE;
                ElemType tolerance = (rangeHigh - rangeLow) * PRECISION_TOLERANCE;

                // First verify that (cpuInMatrix + cpuPrevResidualMatrix + cpuPrevOutMatrix == gpuNewResidualMatrix + gpuNewOutMatrix)
                size_t numMatrixElems = inMatrix.GetNumElements();
                for (size_t i = 0; i < numMatrixElems; ++i)
                {
                    Assert::IsTrue(fabs((gpuInMatrix[i] + gpuPrevResidualMatrix[i] + gpuPrevOutMatrix[i]) - (gpuNewResidualMatrix[i] + gpuNewOutMatrix[i])) <= tolerance);
                }

                // Now verify against the reference CPU quantizer
                ElemType* refNewOutMatrix = new ElemType[numMatrixElems];
                ElemType* refNewResidualMatrix = new ElemType[numMatrixElems];
                ReferenceCPU1BitQuantizer(numRows, numCols, gpuInMatrix, gpuPrevResidualMatrix, gpuPrevOutMatrix, refNewOutMatrix, refNewResidualMatrix);
                for (size_t i = 0; i < numMatrixElems; ++i)
                {
                    Assert::IsTrue(fabs(gpuNewOutMatrix[i] - refNewOutMatrix[i]) <= tolerance);
                    Assert::IsTrue(fabs(gpuNewResidualMatrix[i] - refNewResidualMatrix[i]) <= tolerance);
                }

                delete[] gpuInMatrix;
                delete[] gpuPrevResidualMatrix;
                delete[] gpuPrevOutMatrix;
                delete[] gpuNewResidualMatrix;
                delete[] gpuNewOutMatrix;
                delete[] refNewOutMatrix;
                delete[] refNewResidualMatrix;
            }

            delete quantizer;
            delete allocator;
        }

        template <typename ElemType>
        static void Test1BitQuantization(short deviceId)
        {
            size_t numRows = 256;
            size_t numCols = 135;
            float rangeLow = -1.0f;
            float rangeHigh = 1.0f;
            int seed = 2015;
            int numIterations = 5;

            // Test 1-bit quantization on a matrix of size 1024 * 1812 initialized with floating point numbers between -1 and + 1
            Test1BitQuantization<ElemType>(numRows, numCols, rangeLow, rangeHigh, seed, numIterations, deviceId);

            // Test a matrix with smaller range of values
            seed += 100;
            rangeLow = -0.005f;
            rangeHigh = 0.005f;
            Test1BitQuantization<ElemType>(numRows, numCols, rangeLow, rangeHigh, seed, numIterations, deviceId);

            // Test a matrix with larger range of values
            seed += 100;
            rangeLow = -10.0f;
            rangeHigh = 10.0f;
            Test1BitQuantization<ElemType>(numRows, numCols, rangeLow, rangeHigh, seed, numIterations, deviceId);

            // Test a matrix with assymmetric range of values
            seed += 100;
            rangeLow = -1.0f;
            rangeHigh = 2.05f;
            Test1BitQuantization<ElemType>(numRows, numCols, rangeLow, rangeHigh, seed, numIterations, deviceId);

            // Test a matrix with a single column
            seed += 100;
            rangeLow = -0.5f;
            rangeHigh = 0.5f;
            numRows = 489;
            numCols = 1;
            Test1BitQuantization<ElemType>(numRows, numCols, rangeLow, rangeHigh, seed, numIterations, deviceId);

            // Test a matrix with a single row
            seed += 100;
            rangeLow = -0.5f;
            rangeHigh = 0.5f;
            numRows = 1;
            numCols = 135;
            Test1BitQuantization<ElemType>(numRows, numCols, rangeLow, rangeHigh, seed, numIterations, deviceId);

            // Test a matrix with a number of rows that is not a multiple of the number of bits in a quantized word
            seed += 100;
            rangeLow = -0.5f;
            rangeHigh = 0.5f;
            numRows = 89;
            numCols = 23;
            Test1BitQuantization<ElemType>(numRows, numCols, rangeLow, rangeHigh, seed, numIterations, deviceId);

            // Test a matrix with a number of rows less than number of bits in a quantized word
            seed += 100;
            rangeLow = -0.5f;
            rangeHigh = 0.5f;
            numRows = 15;
            numCols = 135;
            Test1BitQuantization<ElemType>(numRows, numCols, rangeLow, rangeHigh, seed, numIterations, deviceId);

            // Test with a large matrix
            seed += 100;
            rangeLow = -0.5f;
            rangeHigh = 0.5f;
            numRows = 1537;
            numCols = 973;
            Test1BitQuantization<ElemType>(numRows, numCols, rangeLow, rangeHigh, seed, numIterations, deviceId);
        }

    public:
        //This test will fail without GPU
        TEST_METHOD(Matrix1BitQuantize)
        {
            // Test single precision 1bit quantization on CPU
            Test1BitQuantization<float>(CPUDEVICE);

            // Test double precision 1bit quantization on CPU
            Test1BitQuantization<double>(CPUDEVICE);

            const int GPUDEVICE = 0;

            // Test single precision 1bit quantization on GPU
            Test1BitQuantization<float>(GPUDEVICE);

            // Test double precision 1bit quantization on GPU
            Test1BitQuantization<double>(GPUDEVICE);
        }
    };

    /*static*/ const float MatrixQuantizerTests::SINGLE_PRECISION_TOLERANCE = 0.00001f;
    /*static*/ const double MatrixQuantizerTests::DOUBLE_PRECISION_TOLERANCE = 0.0000000000001;
}

