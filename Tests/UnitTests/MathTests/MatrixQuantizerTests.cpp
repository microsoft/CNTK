//
// <copyright file="MatrixQuantizerTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#include "stdafx.h"
#include "File.h"
#include <memory>
#include <io.h>

#include "../../../Math/Math/MatrixQuantizer.h"
#include "../../../Math/Math/CUDAPageLockedMemAllocator.h"
#include "../../../Math/Math/ValueQuantizer.h"

using namespace Microsoft::MSR::CNTK;

// #define DEBUG_OUTPUT_PATH L"d:/MatrixQuantizerTest.out.txt"

void RedirectStdErrAndStdOut(bool createDebugOut)
{
    if (createDebugOut)
    {
#ifdef DEBUG_OUTPUT_PATH
        wString logPath(DEBUG_OUTPUT_PATH);
#else
        wstring logPath(L"");
#endif
        fprintf(stderr, "Redirecting stderr to file %S\n", logPath.c_str());
        auto f = make_shared<File>(logPath.c_str(), fileOptionsWrite | fileOptionsText);
        if (_dup2(_fileno(*f), 1) == -1)
            RuntimeError("unexpected failure to redirect stdout to log file");
        if (_dup2(_fileno(*f), 2) == -1)
            RuntimeError("unexpected failure to redirect stderr to log file");
        setvbuf(stderr, NULL, _IONBF, 16384);   // unbuffer it
        static auto fKept = f;                  // keep it around (until it gets changed)
    }
}


namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {


#ifdef DEBUG_OUTPUT_PATH
    static bool createDebugOut = true;
#else
    static bool createDebugOut = false;
#endif
    //static const float c_SinglePrecisionTolerance = 0.00005f;
    //static const double c_DoublePrecisionTolerance = 0.000000001;
    //static const float c_SinglePrecisionGpuQuantizationTolerance = 0.0001f;
    static const float c_SinglePrecisionTolerance = 0.0001f;
    static const double c_DoublePrecisionTolerance = 0.00000001;
    static const float c_SinglePrecisionGpuQuantizationTolerance = 0.001f;

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
                for (auto i = 0; i < numRows; i++)
                {
                    auto flatIdx = (j * numRows) + i;
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
                ElemType sum = static_cast<ElemType>(0.0);
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
                    // we minimize the error jointly across positive and negative numbers to make things
                    // symmetrical around the mean (which may be non-zero) tying the two sides
                    ElemType devacc0 = (num0 * mean) - mean0Sum;
                    ElemType devacc1 = mean1Sum - (num1 * mean);

                    // both deviations tied, to ensure consistent mean
                    ElemType dev = (devacc0 + devacc1) / numRows;
                    radius = static_cast<ElemType>(2.0) * dev;
                    newMean = mean;
                }
                else
                {
                    // happens for all-zero columns which do exist (mean0 is 0 in that case)
                    if (num0 == 0)
                    {
                        num0 = 1;
                    }
                    if (num1 == 0)
                    {
                        num1 = 1;
                    }

                    const ElemType mean0 = mean0Sum / num0;
                    const ElemType mean1 = mean1Sum / num1;

                    newMean = static_cast<ElemType>(0.5) * (mean0 + mean1);
                    radius = static_cast<ElemType>(2.0) * (mean1 - newMean);
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
                    const size_t flatIdx = (j * numRows) + i;
                    ElemType val = inMatrix[flatIdx] + prevResidualMatrix[flatIdx];
                    varacc += (val - mean) * (val - mean);
                }

                ElemType stddev = sqrt(varacc / numRows);
                quantiMin = mean - (stddevs * stddev);
                quantiMax = mean + (stddevs * stddev);
            }

            ElemType qFactor;
            ElemType uFactor;
            QWordVal rangeSize = (static_cast<QWordVal> (1)) << numBits;

            // must protect against NaN: interval is 0 -> quantization is futile, just emit 0
            if (((quantiMax - quantiMin) < 1e-36f) || (rangeSize == 0))
            {
                qFactor = uFactor = static_cast<ElemType>(0.0);
            }
            else
            {
                qFactor = rangeSize / (quantiMax - quantiMin);
                uFactor = (quantiMax - quantiMin) / rangeSize;
            }

            for (int i = 0; i < numRows; i++)
            {
                auto flatIdx = (j * numRows) + i;
                ElemType val = inMatrix[flatIdx] + prevResidualMatrix[flatIdx];
                ElemType qVal;

                if (numBits == 1)
                {
                    if (val < mean)
                    {
                        qVal = newMean - (static_cast<ElemType>(0.5) * radius);
                    }
                    else
                    {
                        qVal = newMean + (static_cast<ElemType>(0.5) * radius);
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
                        result = static_cast<QWordValSigned>(rangeSize - 1);
                    }
                    else
                    {
                        result = static_cast<QWordValSigned>((val - quantiMin) * qFactor);
                    }

                    qVal = ((static_cast<QWordVal> (result) + static_cast<ElemType>(0.5)) * uFactor) + quantiMin;
                }

                outMatrix[flatIdx] = prevOutMatrix[flatIdx] + qVal;
                outResidualMatrix[flatIdx] = val - qVal;
            }
        }
    }

    template <typename ElemType>
    static void TestRunQuantization(
        size_t numBits,
        size_t numRows,
        size_t numCols,
        ElemType rangeLow,
        ElemType rangeHigh,
        int seed,
        int numIterations,
        int deviceId,
        bool zeroThresholdFor1Bit)
    {
        auto verifyAllZerosFunc = [](const Matrix<ElemType>& matrix) 
        {
            std::unique_ptr<ElemType[]> cpuMatrix(matrix.CopyToArray());
            const size_t numMatrixElems = matrix.GetNumElements();

            for (size_t i = 0; i < numMatrixElems; ++i)
            {
                BOOST_CHECK_EQUAL(cpuMatrix[i], static_cast<ElemType>(0));
            }
        };
    
        std::unique_ptr<MemAllocator> allocator(deviceId == CPUDEVICE ? nullptr : new CUDAPageLockedMemAllocator(deviceId));
    
        Matrix<ElemType> inMatrix(numRows, numCols, deviceId);
        std::unique_ptr<MatrixQuantizer<ElemType>> quantizer(MatrixQuantizer<ElemType>::CreateMatrixQuantizer(numRows, numCols, deviceId));

        // Verify that the initial residue is comprised of all zeros
        verifyAllZerosFunc(quantizer->GetResidualMatrix());
        Matrix<ElemType> outMatrix(numRows, numCols, deviceId);
        // Verify that the outMatrix is initialized with all zeros
        verifyAllZerosFunc(outMatrix);

        for (int iterNum = 0; iterNum < numIterations; ++iterNum)
        {
            inMatrix = Matrix<ElemType>::RandomUniform(numRows, numCols, rangeLow, rangeHigh, seed + iterNum, deviceId);

            std::unique_ptr<ElemType[]> gpuInMatrix(inMatrix.CopyToArray());
            std::unique_ptr<ElemType[]> gpuPrevResidualMatrix(quantizer->GetResidualMatrix().CopyToArray());
            std::unique_ptr<ElemType[]> gpuPrevOutMatrix(outMatrix.CopyToArray());

            size_t numRowsToPrint(0);
            size_t numColsToPrint(0);
            if (createDebugOut)
            {
                bool peekOnly = true;
                const size_t numRowsToPeek = 3;
                const size_t numColsToPeek = 3;
                if (peekOnly)
                {
                    numRowsToPrint = (std::min)(numRowsToPeek, numRows);
                    numColsToPrint = (std::min)(numColsToPeek, numCols);
                }
                else
                {
                    numRowsToPrint = numRows;
                    numColsToPrint = numCols;
                }

                inMatrix.Print("Input Matrix", 0, numRowsToPrint - 1, 0, numColsToPrint - 1);
                quantizer->GetResidualMatrix().Print("Old Residual Matrix", 0, numRowsToPrint - 1, 0, numColsToPrint - 1);
                outMatrix.Print("Old Output Matrix", 0, numRowsToPrint - 1, 0, numColsToPrint - 1);
            }

            QuantizedMatrix<ElemType> tempCPUQuantizationBuffer(numRows, numCols, numBits, CPUDEVICE, allocator.get());
            quantizer->QuantizeAsync(inMatrix, tempCPUQuantizationBuffer, zeroThresholdFor1Bit);
            quantizer->WaitQuantizeAsyncDone();

            if (createDebugOut)
            {
                tempCPUQuantizationBuffer.Print("Quantized Matrix", 0, numRowsToPrint - 1, 0, numColsToPrint - 1);
                quantizer->GetResidualMatrix().Print("New residual Matrix", 0, numRowsToPrint - 1, 0, numColsToPrint - 1);
            }

            quantizer->UnquantizeAsync(tempCPUQuantizationBuffer, outMatrix, (iterNum > 0));
            quantizer->WaitUnquantizeAsyncDone();

            if (createDebugOut)
            {
                outMatrix.Print("Unquantized Output Matrix", 0, numRowsToPrint - 1, 0, numColsToPrint - 1);
            }

            // Now verify the quantization results
            std::unique_ptr<ElemType[]> gpuNewResidualMatrix(quantizer->GetResidualMatrix().CopyToArray());
            std::unique_ptr<ElemType[]> gpuNewOutMatrix(outMatrix.CopyToArray());

            ElemType precisionTolerance = (std::is_same<ElemType, double>::value) ? static_cast<ElemType>(c_DoublePrecisionTolerance) : c_SinglePrecisionTolerance;
            ElemType tolerance = 0.0f;
            if (numBits != (8 * sizeof(ElemType)))
            {
                tolerance = (rangeHigh - rangeLow) * precisionTolerance;
            }
            
            const size_t numMatrixElems = inMatrix.GetNumElements();
            for (size_t i = 0; i < numMatrixElems; ++i)
            {
                BOOST_CHECK(fabs((gpuInMatrix[i] + gpuPrevResidualMatrix[i] + gpuPrevOutMatrix[i]) - (gpuNewResidualMatrix[i] + gpuNewOutMatrix[i])) <= tolerance);
            }

            size_t numIncorrectAllowed = 0;
            if (std::is_same<ElemType, float>::value && (deviceId >= 0))
            {
                // We allow a small number of incorrect results when computing on the GPU
                // for single precision since, in rare cases, the value of the CPU and GPU
                // may quantize to different integers resulting in difference larger than 
                // what is allowed by tolerance
                numIncorrectAllowed = (std::max)(static_cast<size_t>(1), static_cast<size_t>(numMatrixElems * c_SinglePrecisionGpuQuantizationTolerance));
            }

            // Now verify against the reference CPU quantizer
            size_t numIncorrectOutValue = 0;
            size_t numIncorrectResidualValue = 0;
            std::unique_ptr<ElemType[]> refNewOutMatrix(new ElemType[numMatrixElems]);
            std::unique_ptr<ElemType[]> refNewResidualMatrix(new ElemType[numMatrixElems]);

            ReferenceCPUQuantizer(numBits, numRows, numCols, 
                gpuInMatrix.get(), gpuPrevResidualMatrix.get(), gpuPrevOutMatrix.get(), refNewOutMatrix.get(), refNewResidualMatrix.get(), zeroThresholdFor1Bit);
            for (size_t i = 0; i < numMatrixElems; ++i)
            {
                if (fabs(gpuNewOutMatrix[i] - refNewOutMatrix[i]) > tolerance)
                {
                    numIncorrectOutValue++;
                    if (numIncorrectOutValue > numIncorrectAllowed)
                    {
                        BOOST_CHECK_LE(fabs(gpuNewOutMatrix[i] - refNewOutMatrix[i]), tolerance);
                    }
                }

                if (fabs(gpuNewResidualMatrix[i] - refNewResidualMatrix[i]) > tolerance)
                {
                    numIncorrectResidualValue++;
                    if (numIncorrectResidualValue > numIncorrectAllowed)
                    {
                        BOOST_CHECK_LE(fabs(gpuNewResidualMatrix[i] - refNewResidualMatrix[i]), tolerance);
                    }
                }
            }
        }
    }

    template <typename ElemType>
    static void TestQuantization(int deviceId, size_t numRows, size_t numCols, float rangeLow, float rangeHigh, int seed, int numIterations)
    {
        // Test quantization for all power of 2 bit sizes
        const auto maxNumBits = 8 * sizeof(ElemType);
        for (size_t numBits = 1; numBits <= maxNumBits; numBits = numBits * 2)
        {
            // Test 1 bit quantization both with and without zeroThresholdFor1Bit setting
            for (auto i = 0; i < 2; ++i)
            {
                auto zeroThresholdFor1Bit = (i == 1);

                // zeroThresholdFor1Bit test applicable only for 1 bit
                if ((numBits != 1) && zeroThresholdFor1Bit)
                {
                    continue;
                }

                TestRunQuantization<ElemType>(numBits, numRows, numCols, rangeLow, rangeHigh, seed, numIterations, deviceId, zeroThresholdFor1Bit);
            }
        }
    }

    BOOST_AUTO_TEST_SUITE(GPUMatrixSuite);

    BOOST_AUTO_TEST_CASE(GPUMatrix1BitQuantizeFloat)
    {
        RedirectStdErrAndStdOut(createDebugOut);

        TestQuantization<float>(c_deviceIdZero, 25, 13, -1.0f, +1.0f, 2015, 5);
        TestQuantization<float>(c_deviceIdZero, 25, 13, -0.005f, +0.005f, 2115, 5);
        TestQuantization<float>(c_deviceIdZero, 13, 25, -0.00001f, +0.00001f, 2215, 5);
        TestQuantization<float>(c_deviceIdZero, 13, 25, -10.0f, +10.0f, 2315, 5);
        TestQuantization<float>(c_deviceIdZero, 25, 13, -1.0f, +2.05f, 2415, 5);
        TestQuantization<float>(c_deviceIdZero, 489, 1, -0.5f, +0.5f, 2515, 5);
        TestQuantization<float>(c_deviceIdZero, 1, 135, -0.5f, +0.5f, 2615, 5);
        TestQuantization<float>(c_deviceIdZero, 89, 23, -0.5f, +0.5f, 2715, 5);
        TestQuantization<float>(c_deviceIdZero, 15, 35, -0.5f, +0.5f, 2815, 5);
        TestQuantization<float>(c_deviceIdZero, 100, 50, -0.5f, +0.5f, 2915, 5);
    }

    BOOST_AUTO_TEST_CASE(GPUMatrix1BitQuantizeDouble)
    {
        RedirectStdErrAndStdOut(createDebugOut);

        TestQuantization<double>(c_deviceIdZero, 25, 13, -1.0f, +1.0f, 2015, 5);
        TestQuantization<double>(c_deviceIdZero, 25, 13, -0.005f, +0.005f, 2115, 5);
        TestQuantization<double>(c_deviceIdZero, 13, 25, -0.00001f, +0.00001f, 2215, 5);
        TestQuantization<double>(c_deviceIdZero, 13, 25, -10.0f, +10.0f, 2315, 5);
        TestQuantization<double>(c_deviceIdZero, 25, 13, -1.0f, +2.05f, 2415, 5);
        TestQuantization<double>(c_deviceIdZero, 489, 1, -0.5f, +0.5f, 2515, 5);
        TestQuantization<double>(c_deviceIdZero, 1, 135, -0.5f, +0.5f, 2615, 5);
        TestQuantization<double>(c_deviceIdZero, 89, 23, -0.5f, +0.5f, 2715, 5);
        TestQuantization<double>(c_deviceIdZero, 15, 35, -0.5f, +0.5f, 2815, 5);
        TestQuantization<double>(c_deviceIdZero, 100, 50, -0.5f, +0.5f, 2915, 5);
    }

    BOOST_AUTO_TEST_SUITE_END()

    BOOST_AUTO_TEST_SUITE(CPUMatrixSuite);

    BOOST_AUTO_TEST_CASE(CPUMatrix1BitQuantizeFloat)
    {
        RedirectStdErrAndStdOut(createDebugOut);

        TestQuantization<float>(CPUDEVICE, 25, 13, -1.0f, +1.0f, 2015, 5);
        TestQuantization<float>(CPUDEVICE, 25, 13, -0.005f, +0.005f, 2115, 5);
        TestQuantization<float>(CPUDEVICE, 13, 25, -0.00001f, +0.00001f, 2215, 5);
        TestQuantization<float>(CPUDEVICE, 13, 25, -10.0f, +10.0f, 2315, 5);
        TestQuantization<float>(CPUDEVICE, 25, 13, -1.0f, +2.05f, 2415, 5);
        TestQuantization<float>(CPUDEVICE, 489, 1, -0.5f, +0.5f, 2515, 5);
        TestQuantization<float>(CPUDEVICE, 1, 135, -0.5f, +0.5f, 2615, 5);
        TestQuantization<float>(CPUDEVICE, 89, 23, -0.5f, +0.5f, 2715, 5);
        TestQuantization<float>(CPUDEVICE, 15, 35, -0.5f, +0.5f, 2815, 5);
        TestQuantization<float>(CPUDEVICE, 100, 50, -0.5f, +0.5f, 2915, 5);
    }

    BOOST_AUTO_TEST_CASE(CPUMatrix1BitQuantizeDouble)
    {
        RedirectStdErrAndStdOut(createDebugOut);

        TestQuantization<double>(CPUDEVICE, 25, 13, -1.0f, +1.0f, 2015, 5);
        TestQuantization<double>(CPUDEVICE, 25, 13, -0.005f, +0.005f, 2115, 5);
        TestQuantization<double>(CPUDEVICE, 13, 25, -0.00001f, +0.00001f, 2215, 5);
        TestQuantization<double>(CPUDEVICE, 13, 25, -10.0f, +10.0f, 2315, 5);
        TestQuantization<double>(CPUDEVICE, 25, 13, -1.0f, +2.05f, 2415, 5);
        TestQuantization<double>(CPUDEVICE, 489, 1, -0.5f, +0.5f, 2515, 5);
        TestQuantization<double>(CPUDEVICE, 1, 135, -0.5f, +0.5f, 2615, 5);
        TestQuantization<double>(CPUDEVICE, 89, 23, -0.5f, +0.5f, 2715, 5);
        TestQuantization<double>(CPUDEVICE, 15, 35, -0.5f, +0.5f, 2815, 5);
        TestQuantization<double>(CPUDEVICE, 100, 50, -0.5f, +0.5f, 2915, 5);
    }

        /*
        Original test cases were using these parameter:

        TestQuantization<'float or double'>('CPU or GPU', 256, 135, -1.0f, +1.0f, 2015, 5);
        TestQuantization<'float or double'>('CPU or GPU', 256, 135, -0.005f, +0.005f, 2115, 5);
        TestQuantization<'float or double'>('CPU or GPU', 256, 135, -0.00001f, +0.00001f, 2215, 5);
        TestQuantization<'float or double'>('CPU or GPU', 256, 135, -10.0f, +10.0f, 2315, 5);
        TestQuantization<'float or double'>('CPU or GPU', 256, 135, -1.0f, +2.05f, 2415, 5);
        TestQuantization<'float or double'>('CPU or GPU', 489, 1, -0.5f, +0.5f, 2515, 5);
        TestQuantization<'float or double'>('CPU or GPU', 1, 135, -0.5f, +0.5f, 2615, 5);
        TestQuantization<'float or double'>('CPU or GPU', 89, 23, -0.5f, +0.5f, 2715, 5);
        TestQuantization<'float or double'>('CPU or GPU', 15, 135, -0.5f, +0.5f, 2815, 5);
        TestQuantization<'float or double'>('CPU or GPU', 737, 373, -0.5f, +0.5f, 2915, 5);
        */

    BOOST_AUTO_TEST_SUITE_END()

}  }  }  } 

