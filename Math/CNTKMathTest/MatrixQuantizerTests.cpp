//
// <copyright file="CPUMatrixUnitTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#include "stdafx.h"
#include "CppUnitTest.h"
#include "File.h"
#include <memory>
#include <io.h>
#include <type_traits>

#include "..\Math\MatrixQuantizer.h"
#include "..\Math\CUDAPageLockedMemAllocator.h"
#include "..\Math\ValueQuantizer.h"

#define DEBUG_FLAG 1
using namespace Microsoft::MSR::CNTK;

#pragma warning (disable: 4305)

using namespace Microsoft::MSR::CNTK;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

//#define DEBUG_OUTPUT_PATH L"E:/temp/MatrixQuantizerTest.out.txt"

#pragma warning (disable: 4996)

void RedirectStdErrAndStdOut(wstring logpath)
{
    fprintf(stderr, "Redirecting stderr to file %S\n", logpath.c_str());
    auto f = make_shared<File>(logpath.c_str(), fileOptionsWrite | fileOptionsText);
    if (dup2(fileno(*f), 1) == -1)
        RuntimeError("unexpected failure to redirect stdout to log file");
    if (dup2(fileno(*f), 2) == -1)
        RuntimeError("unexpected failure to redirect stderr to log file");
    setvbuf(stderr, NULL, _IONBF, 16384);   // unbuffer it
    static auto fKept = f;                  // keep it around (until it gets changed)
}

namespace CNTKMathTest
{
    TEST_CLASS(MatrixQuantizerTests)
    {
    private:

        static const float SINGLE_PRECISION_TOLERANCE;
        static const double DOUBLE_PRECISION_TOLERANCE;

        template <typename ElemType>
        static void ReferenceCPUQuantizer(
            size_t numBits,
            size_t numRows,
            size_t numCols,
            const ElemType* inMatrix,
            const ElemType* prevResidualMatrix,
            const ElemType* prevOutMatrix,
            ElemType* outMatrix,
            ElemType* outResidualMatrix,
            bool zeroThresholdFor1Bit)
        {
            typedef typename QuantizedWordHelper<ElemType>::ValueType QWordVal;
            typedef typename QuantizedWordHelper<ElemType>::ValueTypeSigned QWordValSigned;

            // Just pass through the values if numBits is of the full size of the ElemType
            if (numBits == (8 * sizeof(ElemType)))
            {
                for (size_t j = 0; j < numCols; j++)
                {
                    for (int i = 0; i < numRows; i++)
                    {
                        size_t flatIdx = (j * numRows) + i;
                        ElemType val = inMatrix[flatIdx] + prevResidualMatrix[flatIdx];
                        outMatrix[flatIdx] = prevOutMatrix[flatIdx] + val;
                        outResidualMatrix[flatIdx] = 0;
                    }
                }

                return;
            }

            for (size_t j = 0; j < numCols; j++)
            {
                ElemType mean = 0.0f;
                if (!zeroThresholdFor1Bit || (numBits != 1))
                {
                    ElemType sum = (ElemType)0.0;
                    for (int i = 0; i < numRows; i++)
                    {
                        size_t flatIdx = (j * numRows) + i;
                        sum += inMatrix[flatIdx] + prevResidualMatrix[flatIdx];
                    }
                    mean = sum / numRows;
                }

                ElemType radius = 0.0f;
                ElemType newMean = 0.0f;
                ElemType quantiMin;
                ElemType quantiMax;
                if (numBits == 1)
                {
                    // Calculate the mean0 and mean1 for each column
                    ElemType mean0Sum = 0.0f;
                    ElemType mean1Sum = 0.0f;
                    int num0 = 0;
                    int num1 = 0;
                    for (int i = 0; i < numRows; i++)
                    {
                        size_t flatIdx = (j * numRows) + i;
                        ElemType val = inMatrix[flatIdx] + prevResidualMatrix[flatIdx];
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

                    if (!zeroThresholdFor1Bit)
                    {
                        // we minimize the error jointly across positive and negative numbers to make things symmetrical around the mean (which may be non-zero)
                        // tying the two sides
                        ElemType devacc0 = (num0 * mean) - mean0Sum;
                        ElemType devacc1 = mean1Sum - (num1 * mean);

                        // both deviations tied, to ensure consistent mean
                        ElemType dev = (devacc0 + devacc1) / numRows;
                        radius = (ElemType)2.0 * dev;
                        newMean = mean;
                    }
                    else
                    {
                        if (num0 == 0) num0 = 1;                        // happens for all-zero columns which do exist (mean0 is 0 in that case)
                        if (num1 == 0) num1 = 1;
                        const ElemType mean0 = mean0Sum / num0;
                        const ElemType mean1 = mean1Sum / num1;

                        newMean = (ElemType)0.5 * (mean0 + mean1);
                        radius = (ElemType)2.0 * (mean1 - newMean);
                    }

                    quantiMin = newMean - radius;
                    quantiMax = newMean + radius;
                }
                else
                {
                    // >1 bit:
                    // We linearly quantize between 'stddevs' standard deviations.
                    ElemType stddevs = 5.0f;
                    ElemType varacc = 0.0f;
                    for (int i = 0; i < numRows; i++)
                    {
                        size_t flatIdx = (j * numRows) + i;
                        ElemType val = inMatrix[flatIdx] + prevResidualMatrix[flatIdx];
                        varacc += (val - mean) * (val - mean);
                    }

                    ElemType stddev = sqrt(varacc / numRows);
                    quantiMin = mean - (stddevs * stddev);
                    quantiMax = mean + (stddevs * stddev);
                }

                ElemType qFactor;
                ElemType uFactor;
                QWordVal rangeSize = 1 << numBits;

                // must protect against NaN: interval is 0 -> quantization is futile, just emit 0
                if (((quantiMax - quantiMin) < 1e-36f) || (rangeSize == 0))
                {
                    qFactor = uFactor = (ElemType)0.0;
                }
                else
                {
                    qFactor = rangeSize / (quantiMax - quantiMin);
                    uFactor = (quantiMax - quantiMin) / rangeSize;
                }

                for (int i = 0; i < numRows; i++)
                {
                    size_t flatIdx = (j * numRows) + i;
                    ElemType val = inMatrix[flatIdx] + prevResidualMatrix[flatIdx];
                    ElemType qVal;

                    if (numBits == 1)
                    {
                        if (val < mean)
                        {
                            qVal = newMean - ((ElemType)0.5 * radius);
                        }
                        else
                        {
                            qVal = newMean + ((ElemType)0.5 * radius);
                        }
                    }
                    else
                    {
                        QWordValSigned result;
                        if (val <= quantiMin)
                        {
                            result = 0;
                        }
                        else if (val >= quantiMax)
                        {
                            result = (QWordValSigned)(rangeSize - 1);
                        }
                        else
                        {
                            result = (QWordValSigned)((val - quantiMin) * qFactor);
                        }

                        qVal = (((QWordVal)result + (ElemType)0.5) * uFactor) + quantiMin;
                    }

                    outMatrix[flatIdx] = prevOutMatrix[flatIdx] + qVal;
                    outResidualMatrix[flatIdx] = val - qVal;
                }
            }
        }

        template <typename ElemType>
        static void TestQuantization(
            size_t numBits,
            size_t numRows,
            size_t numCols,
            ElemType rangeLow,
            ElemType rangeHigh,
            int seed,
            int numIterations,
            short deviceId,
            bool zeroThresholdFor1Bit)
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

#ifdef DEBUG_OUTPUT_PATH
                const size_t numRowsToPrint = 3;
                const size_t numColsToPrint = 3;
                inMatrix.Print("Input Matrix", 0, numRowsToPrint - 1, 0, numColsToPrint - 1);
                quantizer->GetResidualMatrix().Print("Old Residual Matrix", 0, numRowsToPrint - 1, 0, numColsToPrint - 1);
                outMatrix.Print("Old Output Matrix", 0, numRowsToPrint - 1, 0, numColsToPrint - 1);
#endif

                QuantizedMatrix<ElemType> tempCPUQuantizationBuffer(numRows, numCols, numBits, CPUDEVICE, allocator);
                quantizer->QuantizeAsync(tempCPUQuantizationBuffer, zeroThresholdFor1Bit);
                quantizer->WaitQuantizeAsyncDone();

#ifdef DEBUG_OUTPUT_PATH
                tempCPUQuantizationBuffer.Print("Quantized Matrix", 0, numRowsToPrint - 1, 0, numColsToPrint - 1);
                quantizer->GetResidualMatrix().Print("New residual Matrix", 0, numRowsToPrint - 1, 0, numColsToPrint - 1);
#endif

                quantizer->UnquantizeAsync(tempCPUQuantizationBuffer, outMatrix, (iterNum > 0));
                quantizer->WaitUnquantizeAsyncDone();

#ifdef DEBUG_OUTPUT_PATH
                outMatrix.Print("Unquantized Output Matrix", 0, numRowsToPrint - 1, 0, numColsToPrint - 1);
#endif

                // Now verify the quantization results
                ElemType* gpuNewResidualMatrix = quantizer->GetResidualMatrix().CopyToArray();
                ElemType* gpuNewOutMatrix = outMatrix.CopyToArray();

                ElemType PRECISION_TOLERANCE = (std::is_same<ElemType, double>::value) ? ((ElemType)DOUBLE_PRECISION_TOLERANCE) : SINGLE_PRECISION_TOLERANCE;
                ElemType tolerance = 0.0f;
                if (numBits != (8 * sizeof(ElemType)))
                {
                    tolerance = (rangeHigh - rangeLow) * PRECISION_TOLERANCE;
                }

                // First verify that (cpuInMatrix + cpuPrevResidualMatrix + cpuPrevOutMatrix == gpuNewResidualMatrix + gpuNewOutMatrix)
                size_t numMatrixElems = inMatrix.GetNumElements();
                for (size_t i = 0; i < numMatrixElems; ++i)
                {
                    Assert::IsTrue(fabs((gpuInMatrix[i] + gpuPrevResidualMatrix[i] + gpuPrevOutMatrix[i]) - (gpuNewResidualMatrix[i] + gpuNewOutMatrix[i])) <= tolerance);
                }

                // Now verify against the reference CPU quantizer
                ElemType* refNewOutMatrix = new ElemType[numMatrixElems];
                ElemType* refNewResidualMatrix = new ElemType[numMatrixElems];
                ReferenceCPUQuantizer(numBits, numRows, numCols, gpuInMatrix, gpuPrevResidualMatrix, gpuPrevOutMatrix, refNewOutMatrix, refNewResidualMatrix, zeroThresholdFor1Bit);
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
        static void TestQuantization(short deviceId)
        {
            // Test quantization for all power of 2 bit sizes
            const size_t maxNumBits = 8 * sizeof(ElemType);
            for (size_t numBits = 1; numBits <= maxNumBits; numBits = numBits * 2)
            {
                // Test 1 bit quantization both with and without zeroThresholdFor1Bit setting
                for (int i = 0; i < 2; ++i)
                {
                    bool zeroThresholdFor1Bit = (i == 1);

                    // zeroThresholdFor1Bit test applicable only for 1 bit
                    if ((numBits != 1) && zeroThresholdFor1Bit)
                    {
                        continue;
                    }

                    // Test quantization on a matrix initialized with floating point numbers between -1 and + 1
                    size_t numRows = 256;
                    size_t numCols = 135;
                    float rangeLow = -1.0f;
                    float rangeHigh = 1.0f;
                    int seed = 2015;
                    int numIterations = 5;
                    TestQuantization<ElemType>(numBits, numRows, numCols, rangeLow, rangeHigh, seed, numIterations, deviceId, zeroThresholdFor1Bit);

                    // Test a matrix with smaller range of values
                    seed += 100;
                    rangeLow = -0.005f;
                    rangeHigh = 0.005f;
                    TestQuantization<ElemType>(numBits, numRows, numCols, rangeLow, rangeHigh, seed, numIterations, deviceId, zeroThresholdFor1Bit);

                    // Test a matrix with larger range of values
                    seed += 100;
                    rangeLow = -10.0f;
                    rangeHigh = 10.0f;
                    TestQuantization<ElemType>(numBits, numRows, numCols, rangeLow, rangeHigh, seed, numIterations, deviceId, zeroThresholdFor1Bit);

                    // Test a matrix with assymmetric range of values
                    seed += 100;
                    rangeLow = -1.0f;
                    rangeHigh = 2.05f;
                    TestQuantization<ElemType>(numBits, numRows, numCols, rangeLow, rangeHigh, seed, numIterations, deviceId, zeroThresholdFor1Bit);

                    // Test a matrix with a single column
                    seed += 100;
                    rangeLow = -0.5f;
                    rangeHigh = 0.5f;
                    numRows = 489;
                    numCols = 1;
                    TestQuantization<ElemType>(numBits, numRows, numCols, rangeLow, rangeHigh, seed, numIterations, deviceId, zeroThresholdFor1Bit);

                    // Test a matrix with a single row
                    seed += 100;
                    rangeLow = -0.5f;
                    rangeHigh = 0.5f;
                    numRows = 1;
                    numCols = 135;
                    TestQuantization<ElemType>(numBits, numRows, numCols, rangeLow, rangeHigh, seed, numIterations, deviceId, zeroThresholdFor1Bit);

                    // Test a matrix with a number of rows that is not a multiple of the number of bits in a quantized word
                    seed += 100;
                    rangeLow = -0.5f;
                    rangeHigh = 0.5f;
                    numRows = 89;
                    numCols = 23;
                    TestQuantization<ElemType>(numBits, numRows, numCols, rangeLow, rangeHigh, seed, numIterations, deviceId, zeroThresholdFor1Bit);

                    // Test a matrix with a number of rows less than number of bits in a quantized word
                    seed += 100;
                    rangeLow = -0.5f;
                    rangeHigh = 0.5f;
                    numRows = 15;
                    numCols = 135;
                    TestQuantization<ElemType>(numBits, numRows, numCols, rangeLow, rangeHigh, seed, numIterations, deviceId, zeroThresholdFor1Bit);

                    // Test with a large matrix
                    seed += 100;
                    rangeLow = -0.5f;
                    rangeHigh = 0.5f;
                    numRows = 737;
                    numCols = 373;
                    TestQuantization<ElemType>(numBits, numRows, numCols, rangeLow, rangeHigh, seed, numIterations, deviceId, zeroThresholdFor1Bit);
                }
            }
        }

    public:
        //This test will fail without GPU
        TEST_METHOD(Matrix1BitQuantize)
        {
#ifdef DEBUG_OUTPUT_PATH
            RedirectStdErrAndStdOut(DEBUG_OUTPUT_PATH);
#endif
            const int GPUDEVICE = 0;

            // Test double precision 1bit quantization on GPU
            TestQuantization<double>(GPUDEVICE);

            // Test single precision 1bit quantization on GPU
            TestQuantization<float>(GPUDEVICE);

            // Test double precision 1bit quantization on CPU
            TestQuantization<double>(CPUDEVICE);

            // Test single precision 1bit quantization on CPU
            TestQuantization<float>(CPUDEVICE);
        }
    };

    /*static*/ const float MatrixQuantizerTests::SINGLE_PRECISION_TOLERANCE = 0.00001f;
    /*static*/ const double MatrixQuantizerTests::DOUBLE_PRECISION_TOLERANCE = 0.0000000000001;
}
