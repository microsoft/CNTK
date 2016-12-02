//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include <vector>
#include "CNTKLibrary.h"
#include "Common.h"

using namespace CNTK;
using namespace std;

template <typename ElementType>
void ValueCreationNoNDMaskTest(const DeviceDescriptor device, bool readOnly)
{
    size_t numberOfSequences = 5;
    size_t seqLen = 4;
    vector<size_t> dims{3, 2};

    vector<vector<ElementType>> data;
    for (int n = 0; n < numberOfSequences; n++)
    {
        auto p = new vector<ElementType>();
        data.push_back(*p);
        for (int s = 0; s < seqLen; s++)
        {
            for (int r = 0; r < dims[1]; r++)
            {
                for (int c = 0; c < dims[0]; c++)
                {
                    data[n].push_back(static_cast<ElementType>(n * 1000 + s * 100 + r * 10 + c));
                }
            }
        }
    }

    NDShape sampleShape(dims);
    ValuePtr testValue = Value::Create(sampleShape, data, device, readOnly);

    if (testValue->Shape().Rank() != dims.size() + 2)
    {
        ReportFailure("Shape rank incorrect. Expected: 4, actual: %d\n", testValue->Shape().Rank());
    }
    int rank = 0;
    for (; rank < dims.size(); rank++)
    {
        if (testValue->Shape()[rank] != dims[rank])
        {
            ReportFailure("Shape rank[%d] incorrect. Expected: %d, actual: %d\n", rank, dims[rank], testValue->Shape()[rank]);
        }
    }
    // The next is the length of sequence
    if (testValue->Shape()[rank] != seqLen)
    {
        ReportFailure("The sequence length does not match. Expected: %d, actual: %d\n", seqLen, testValue->Shape()[rank]);
    }
    rank++;
    // The next is the number of sequences
    if (testValue->Shape()[rank] != numberOfSequences)
    {
        ReportFailure("The number of sequences does not match. Expected: %d, actual: %d\n", numberOfSequences, testValue->Shape()[rank]);
    }

    // Check data is also correct.
    vector<ElementType> outputData(testValue->Shape().TotalSize());
    NDShape outputShape(dims);
    outputShape = outputShape.AppendShape({seqLen, numberOfSequences});
    NDArrayViewPtr arrayOutput = MakeSharedObject<NDArrayView>(outputShape, outputData, false);
    arrayOutput->CopyFrom(*testValue->Data());

    int sIndex = 0;
    int oIndex = 0;
    for (int n = 0; n < numberOfSequences; n++)
    {
        sIndex = 0;
        for (int s = 0; s < seqLen; s++)
        {
            for (int r = 0; r < dims[1]; r++)
            {
                for (int c = 0; c < dims[0]; c++, sIndex++, oIndex++)
                {
                    if (data[n][sIndex] != outputData[oIndex])
                    {
                        ReportFailure("Data does match at position %d, expected: %f, actual: %f\n", oIndex, data[n][sIndex], outputData[oIndex]);
                    }
                }
            }
        }
    }
}

template <typename ElementType>
void ValueCreationWithNDMaskTest(const DeviceDescriptor device, bool readOnly)
{
    size_t numberOfSequences = 4;
    vector<size_t> seqLenList{5, 6, 8, 7};
    vector<size_t> dims{3, 2};
    size_t maxSeqLen = 0;

    vector<vector<ElementType>> data;
    for (int n = 0; n < numberOfSequences; n++)
    {
        auto p = new vector<ElementType>();
        data.push_back(*p);
        size_t seqLen = seqLenList[n];
        maxSeqLen = max(seqLen, maxSeqLen);
        for (int s = 0; s < seqLen; s++)
        {
            for (int r = 0; r < dims[1]; r++)
            {
                for (int c = 0; c < dims[0]; c++)
                {
                    data[n].push_back(static_cast<ElementType>(n * 1000 + s * 100 + r * 10 + c));
                }
            }
        }
    }

    NDShape sampleShape(dims);
    ValuePtr testValue = Value::Create(sampleShape, data, device, readOnly);

    if (testValue->Shape().Rank() != dims.size() + 2)
    {
        ReportFailure("Shape rank incorrect. Expected: 4, actual: %d\n", testValue->Shape().Rank());
    }
    int rank = 0;
    for (; rank < dims.size(); rank++)
    {
        if (testValue->Shape()[rank] != dims[rank])
        {
            ReportFailure("Shape rank[%d] incorrect. Expected: %d, actual: %d\n", rank, dims[rank], testValue->Shape()[rank]);
        }
    }
    // The next is the length of sequence
    if (testValue->Shape()[rank] != maxSeqLen)
    {
        ReportFailure("The sequence length does not match. Expected: %d, actual: %d\n", maxSeqLen, testValue->Shape()[rank]);
    }
    rank++;
    // The next is the number of sequences
    if (testValue->Shape()[rank] != numberOfSequences)
    {
        ReportFailure("The number of sequences does not match. Expected: %d, actual: %d\n", numberOfSequences, testValue->Shape()[rank]);
    }

    // Check data is also correct.
    vector<ElementType> outputData(testValue->Shape().TotalSize());
    NDShape outputShape(dims);
    outputShape = outputShape.AppendShape({maxSeqLen, numberOfSequences});
    NDArrayViewPtr arrayOutput = MakeSharedObject<NDArrayView>(outputShape, outputData, false);
    arrayOutput->CopyFrom(*testValue->Data());

    int sIndex = 0;
    size_t oIndex = 0;
    for (int n = 0; n < numberOfSequences; n++)
    {
        sIndex = 0;
        size_t seqLen = seqLenList[n];
        for (int s = 0; s < seqLen; s++)
        {
            for (int r = 0; r < dims[1]; r++)
            {
                for (int c = 0; c < dims[0]; c++, sIndex++, oIndex++)
                {
                    if (data[n][sIndex] != outputData[oIndex])
                    {
                        ReportFailure("Data does match at position %d, expected: %f, actual: %f\n", oIndex, data[n][sIndex], outputData[oIndex]);
                    }
                }
            }
        }
        // Skip mask data
        oIndex += (maxSeqLen - seqLen) * dims[1] * dims[0];
    }
}

template <typename ElementType>
void ValueCreationOneHotNoNDMaskTest(const DeviceDescriptor device, bool readOnly)
{
    size_t numberOfSequences = 5;
    size_t seqLen = 4;
    size_t vocabSize = 16;

    vector<vector<size_t>> data;
    for (int n = 0; n < numberOfSequences; n++)
    {
        auto p = new vector<size_t>();
        data.push_back(*p);
        for (int s = 0; s < seqLen; s++)
        {
            data[n].push_back((s * 10 + n) % vocabSize);
        }
    }

    ValuePtr testValue = Value::Create<ElementType>(vocabSize, data, device, readOnly);

    if (testValue->Shape().Rank() != 3)
    {
        ReportFailure("Shape rank incorrect. Expected: 3, actual: %d\n", testValue->Shape().Rank());
    }
    if (testValue->Shape()[0] != vocabSize)
    {
        ReportFailure("Shape rank[%d] incorrect. Expected: %d, actual: %d\n", 0, vocabSize, testValue->Shape()[0]);
    }
    // The next is the length of sequence
    if (testValue->Shape()[1] != seqLen)
    {
        ReportFailure("The sequence length does not match. Expected: %d, actual: %d\n", seqLen, testValue->Shape()[1]);
    }
    // The next is the number of sequences
    if (testValue->Shape()[2] != numberOfSequences)
    {
        ReportFailure("The number of sequences does not match. Expected: %d, actual: %d\n", numberOfSequences, testValue->Shape()[2]);
    }

    // Check data is also correct.
    vector<ElementType> outputData(vocabSize * seqLen * numberOfSequences);
    NDShape outputShape({vocabSize, seqLen, numberOfSequences});
    NDArrayViewPtr arrayOutput = MakeSharedObject<NDArrayView>(outputShape, outputData, false);
    arrayOutput->CopyFrom(*testValue->Data());

    int sIndex = 0;
    int oIndex = 0;
    for (int n = 0; n < numberOfSequences; n++)
    {
        sIndex = 0;
        for (int s = 0; s < seqLen; s++, sIndex++)
        {
            for (int c = 0; c < vocabSize; c++, oIndex++)
            {
                if (outputData[oIndex] != 0)
                {
                    if (outputData[oIndex] != 1)
                    {
                        ReportFailure("OneHot vector contains value other than 0 and 1 at seqNo=%d sampleNo=%d position=%d\n", n, s, c);
                    }
                    if (c != data[n][sIndex])
                    {
                        ReportFailure("OneHot Index does match at seqNo=%d, sampleNo=%d, expected: %f, actual: %f\n", n, s, data[n][sIndex], c);
                    }
                }
            }
        }
    }
}

template <typename ElementType>
void ValueCreationOneHotWithNDMaskTest(const DeviceDescriptor device, bool readOnly)
{
    size_t numberOfSequences = 4;
    vector<size_t> seqLenList{5, 6, 8, 7};
    size_t maxSeqLen = 0;
    size_t vocabSize = 13;

    vector<vector<size_t>> data;
    for (int n = 0; n < numberOfSequences; n++)
    {
        auto p = new vector<size_t>();
        data.push_back(*p);
        size_t seqLen = seqLenList[n];
        maxSeqLen = max(seqLen, maxSeqLen);
        for (int s = 0; s < seqLen; s++)
        {
            data[n].push_back((s * 10 + n) % vocabSize);
        }
    }

    ValuePtr testValue = Value::Create<ElementType>(vocabSize, data, device, readOnly);

    if (testValue->Shape().Rank() != 3)
    {
        ReportFailure("Shape rank incorrect. Expected: 3, actual: %d\n", testValue->Shape().Rank());
    }
    if (testValue->Shape()[0] != vocabSize)
    {
        ReportFailure("Shape rank[%d] incorrect. Expected: %d, actual: %d\n", 0, vocabSize, testValue->Shape()[0]);
    }
    // The next is the length of sequence
    if (testValue->Shape()[1] != maxSeqLen)
    {
        ReportFailure("The sequence length does not match. Expected: %d, actual: %d\n", maxSeqLen, testValue->Shape()[1]);
    }
    // The next is the number of sequences
    if (testValue->Shape()[2] != numberOfSequences)
    {
        ReportFailure("The number of sequences does not match. Expected: %d, actual: %d\n", numberOfSequences, testValue->Shape()[2]);
    }

    // Check data is also correct.
    vector<ElementType> outputData(vocabSize * maxSeqLen * numberOfSequences);
    NDShape outputShape({vocabSize, maxSeqLen, numberOfSequences});
    NDArrayViewPtr arrayOutput = MakeSharedObject<NDArrayView>(outputShape, outputData, false);
    arrayOutput->CopyFrom(*testValue->Data());

    size_t sIndex = 0;
    size_t oIndex = 0;
    for (int n = 0; n < numberOfSequences; n++)
    {
        sIndex = 0;
        size_t seqLen = seqLenList[n];
        for (int s = 0; s < seqLen; s++, sIndex++)
        {
            for (int c = 0; c < vocabSize; c++, oIndex++)
            {
                if (outputData[oIndex] != 0)
                {
                    if (outputData[oIndex] != 1)
                    {
                        ReportFailure("OneHot vector contains value other than 0 and 1 at seqNo=%d sampleNo=%d position=%d\n", n, s, c);
                    }
                    if (c != data[n][sIndex])
                    {
                        ReportFailure("OneHot Index does match at seqNo=%d, sampleNo=%d, expected: %f, actual: %f\n", n, s, data[n][sIndex], c);
                    }
                }
            }
        }
        // Skip mask data
        oIndex += (maxSeqLen - seqLen) * vocabSize;
    }
}

void ValueTests()
{
    fprintf(stderr, "\nValueTests..\n");

    ValueCreationNoNDMaskTest<float>(DeviceDescriptor::CPUDevice(), false);
    ValueCreationNoNDMaskTest<double>(DeviceDescriptor::CPUDevice(), true);
    ValueCreationWithNDMaskTest<double>(DeviceDescriptor::CPUDevice(), false);
    ValueCreationWithNDMaskTest<float>(DeviceDescriptor::CPUDevice(), true);
    ValueCreationOneHotNoNDMaskTest<float>(DeviceDescriptor::CPUDevice(), false);
    ValueCreationOneHotNoNDMaskTest<double>(DeviceDescriptor::CPUDevice(), true);
    ValueCreationOneHotWithNDMaskTest<double>(DeviceDescriptor::CPUDevice(), false);
    ValueCreationOneHotWithNDMaskTest<float>(DeviceDescriptor::CPUDevice(), true);
    if (IsGPUAvailable())
    {
        ValueCreationNoNDMaskTest<double>(DeviceDescriptor::GPUDevice(0), false);
        ValueCreationNoNDMaskTest<float>(DeviceDescriptor::GPUDevice(0), true);
        ValueCreationWithNDMaskTest<float>(DeviceDescriptor::GPUDevice(0), false);
        ValueCreationWithNDMaskTest<double>(DeviceDescriptor::GPUDevice(0), true);
        ValueCreationOneHotNoNDMaskTest<double>(DeviceDescriptor::GPUDevice(0), false);
        ValueCreationOneHotNoNDMaskTest<float>(DeviceDescriptor::GPUDevice(0), true);
        ValueCreationOneHotWithNDMaskTest<float>(DeviceDescriptor::GPUDevice(0), false);
        ValueCreationOneHotWithNDMaskTest<double>(DeviceDescriptor::GPUDevice(0), true);
    }
}
