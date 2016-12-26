//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include <vector>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include "CNTKLibrary.h"
#include "Common.h"

using namespace CNTK;
using namespace std;

// Check the actual shape matches the expected shape, sequence number and sample size.
void CheckShape(const NDShape& shape, const NDShape& expectedShape, const size_t expectedNumOfSequences, const size_t expectedSampleSize)
{
    if (shape != expectedShape)
    {
        ReportFailure("The shape of the value does not match. Expected: %S, actual: %S\n", expectedShape.AsString().c_str(), shape.AsString().c_str());
    }
    size_t numOfSequences = shape[shape.Rank() - 1];
    if (numOfSequences != expectedNumOfSequences)
    {
        ReportFailure("The sequence number in the Value does not match. Expected: %" PRIu64 ", actual: %" PRIu64 ".", expectedNumOfSequences, numOfSequences);
    }
    size_t sampleSize = shape.SubShape(0, shape.Rank() - 2).TotalSize();
    if (sampleSize != expectedSampleSize)
    {
        ReportFailure("The sample size in the Value does not match. Expected: %" PRIu64 ", actual: %" PRIu64 ".", expectedSampleSize, sampleSize);
    }
}

// Check the actual Value match the expected shape and the given data (in dense format)
template <typename ElementType>
void CheckValue(const ValuePtr testValue, const NDShape& expectedShape, const size_t expectedSampleSize, const vector<vector<ElementType>>& expectedData, const vector<size_t>& seqLenList)
{
    // Check parameters
    if (expectedData.size() != seqLenList.size())
    {
        ReportFailure("Parameter error: the sequence number in the exepected data and sequence list does not match.");
    }
    for (size_t i = 0; i < expectedData.size(); i++)
    {
        if (expectedData[i].size() != seqLenList[i] * expectedSampleSize)
        {
            ReportFailure("Parameter erroe: the number of data for sequence %" PRIu64 " in the expected data does not match. Expected: %" PRIu64 ", actual: %" PRIu64 ".", i, seqLenList[i] * expectedSampleSize, expectedData[i].size());
        }
    }

    // Check shape 
    CheckShape(testValue->Shape(), expectedShape, seqLenList.size(), expectedSampleSize);

    // Get data from Value
    vector<ElementType> outputData(testValue->Shape().TotalSize());
    NDArrayViewPtr arrayOutput = MakeSharedObject<NDArrayView>(testValue->Shape(), outputData, false);
    arrayOutput->CopyFrom(*testValue->Data());

    size_t maxSeqLen = *max_element(seqLenList.begin(), seqLenList.end());
    size_t oIndex = 0;
    size_t sampleSize = testValue->Shape().SubShape(0, testValue->Shape().Rank() - 2).TotalSize();

    for (size_t seq = 0; seq < seqLenList.size(); seq++)
    {
        size_t seqLen = seqLenList[seq];
        for (size_t sIndex = 0; sIndex < seqLen * sampleSize; sIndex++, oIndex++)
        {
            if (expectedData[seq][sIndex] != outputData[oIndex])
            {
                ReportFailure("Data does match at position %" PRIu64 ", expected: %f, actual: %f\n", oIndex, expectedData[seq][sIndex], outputData[oIndex]);
            }
        }
        // Skip mask data
        oIndex += (maxSeqLen - seqLen) * sampleSize;
    }
}

// Check the actual Value match the expected shape and the given data (in onehot vector format)
template <typename ElementType>
void CheckValue(const ValuePtr testValue, const NDShape& expectedShape, const size_t vocabSize, const vector<vector<size_t>>& expectedData, const vector<size_t>& seqLenList)
{
    // Check parameters
    if (expectedData.size() != seqLenList.size())
    {
        ReportFailure("Parameter error: the sequence number in the exepected data and sequence list does not match.");
    }
    for (size_t i = 0; i < expectedData.size(); i++)
    {
        if (expectedData[i].size() != seqLenList[i])
        {
            ReportFailure("Parameter erroe: the number of data for sequence %" PRIu64 " in the expected data does not match. Expected: %" PRIu64 ", actual: %" PRIu64 ".", i, seqLenList[i], expectedData[i].size());
        }
    }

    // Check shape 
    CheckShape(testValue->Shape(), expectedShape, seqLenList.size(), vocabSize);

    // Get data from Value
    vector<ElementType> outputData(testValue->Shape().TotalSize());
    NDArrayViewPtr arrayOutput = MakeSharedObject<NDArrayView>(testValue->Shape(), outputData, false);
    arrayOutput->CopyFrom(*testValue->Data());

    size_t maxSeqLen = *max_element(seqLenList.begin(), seqLenList.end());
    size_t oIndex = 0;
    for (size_t seq = 0; seq < seqLenList.size(); seq++)
    {
        size_t seqLen = seqLenList[seq];
        for (size_t sample = 0; sample < seqLen; sample++)
        {
            for (size_t c = 0; c < vocabSize; c++, oIndex++)
            {
                if (outputData[oIndex] != 0)
                {
                    if (outputData[oIndex] != 1)
                    {
                        ReportFailure("OneHot vector contains value other than 0 and 1 at seqNo=%" PRIu64 " sampleNo=%" PRIu64 " position=%" PRIu64 "\n", seq, sample, c);
                    }
                    if (c != expectedData[seq][sample])
                    {
                        ReportFailure("OneHot Index does match at seqNo=%" PRIu64 ", sampleNo=%" PRIu64 ", expected: %" PRIu64 ", actual: %" PRIu64 "\n", seq, sample, expectedData[seq][sample], c);
                    }
                }
            }
        }
        // Skip mask data
        oIndex += (maxSeqLen - seqLen) * vocabSize;
    }
}

template <typename ElementType>
void FillDenseMatrixData(vector<vector<ElementType>>& databuf,  const vector<size_t>& seqLenList, const size_t sampleSize)
{
    for (size_t seq = 0; seq < seqLenList.size(); seq++)
    {
        auto p = new vector<ElementType>();
        databuf.push_back(*p);
        size_t seqLen = seqLenList[seq];
        for (size_t sample = 0; sample < seqLen ; sample++)
        {
            for (size_t element = 0; element < sampleSize; element++)
            {
                databuf[seq].push_back(static_cast<ElementType>(seq * 1000 + sample * 100 + element));
            }
        }
    }
}

template <typename ElementType>
void ValueCreationNoNDMaskTest(const DeviceDescriptor device, bool readOnly)
{
    size_t numberOfSequences = 5;
    size_t seqLen = 4;
    vector<size_t> dims{3, 2};

    vector<size_t> seqLenList;
    for (size_t i = 0; i < numberOfSequences; i++)
    {
        seqLenList.push_back(seqLen);
    }
    vector<vector<ElementType>> data;
    FillDenseMatrixData(data, seqLenList, dims[0] * dims[1]);

    // Create the Value object based on the given data and shape.
    NDShape sampleShape(dims);
    ValuePtr testValue = Value::Create(sampleShape, data, device, readOnly);

    // Check whether the created value matches expected shape and data.
    CheckValue(testValue, {dims[0], dims[1], seqLen, numberOfSequences}, dims[0] * dims[1], data, seqLenList);
}

template <typename ElementType>
void ValueCreationWithNDMaskTest(const DeviceDescriptor device, bool readOnly)
{
    vector<size_t> seqLenList{5, 6, 8, 7};
    vector<size_t> dims{3, 2};

    size_t numberOfSequences = seqLenList.size();
    vector<vector<ElementType>> data;
    FillDenseMatrixData(data, seqLenList, dims[0] * dims[1]);

    NDShape sampleShape(dims);
    ValuePtr testValue = Value::Create(sampleShape, data, device, readOnly);

    // Check whether the created value matches expected shape and data.
    size_t maxSeqLen = *max_element(seqLenList.begin(), seqLenList.end());
    CheckValue(testValue, {dims[0], dims[1], maxSeqLen, numberOfSequences}, dims[0] * dims[1], data, seqLenList);
}

template <typename ElementType>
void ValueCreationOneHotNoNDMaskTest(const DeviceDescriptor device, bool readOnly)
{
    size_t numberOfSequences = 5;
    size_t seqLen = 4;
    size_t vocabSize = 16;

    vector<size_t> seqLenList;
    for (size_t i = 0; i < numberOfSequences; i++)
    {
        seqLenList.push_back(seqLen);
    }
    vector<vector<size_t>> data;
    for (size_t n = 0; n < numberOfSequences; n++)
    {
        auto p = new vector<size_t>();
        data.push_back(*p);
        for (size_t s = 0; s < seqLen; s++)
        {
            data[n].push_back((s * 10 + n) % vocabSize);
        }
    }

    ValuePtr testValue = Value::Create<ElementType>(vocabSize, data, device, readOnly);

    CheckValue<ElementType>(testValue, {vocabSize, seqLen, numberOfSequences}, vocabSize, data, seqLenList); 
}

template <typename ElementType>
void ValueCreationOneHotWithNDMaskTest(const DeviceDescriptor device, bool readOnly)
{
    vector<size_t> seqLenList{5, 6, 8, 7};
    size_t maxSeqLen = 0;
    size_t vocabSize = 13;

    vector<vector<size_t>> data;
    size_t numberOfSequences = seqLenList.size();
    for (size_t n = 0; n < numberOfSequences; n++)
    {
        auto p = new vector<size_t>();
        data.push_back(*p);
        size_t seqLen = seqLenList[n];
        maxSeqLen = max(seqLen, maxSeqLen);
        for (size_t s = 0; s < seqLen; s++)
        {
            data[n].push_back((s * 10 + n) % vocabSize);
        }
    }

    ValuePtr testValue = Value::Create<ElementType>(vocabSize, data, device, readOnly);

    CheckValue<ElementType>(testValue, {vocabSize, maxSeqLen, numberOfSequences}, vocabSize, data, seqLenList);
}

void TestSettingParameterValuesManually(const DeviceDescriptor& device)
{
    auto v1_1 = MakeSharedObject<NDArrayView>(0.5, NDShape({ 2, 2 }), device);
    auto v1_2 = MakeSharedObject<NDArrayView>(0.4, NDShape({ 2, 2 }), device);

    Parameter p1(v1_1);
    auto value = p1.Value();

    assert(!AreEqual(v1_1, v1_2) && !AreEqual(p1.Value(), v1_2));

    p1.SetValue(v1_2);
    if (!AreEqual(p1.Value(), v1_2))
        throw std::runtime_error("Parameter value does match the expected value.");

    Parameter p2(CNTK::NDArrayView::RandomUniform<float>({ 10 }, -0.05, 0.05, 1, device));
    auto v2 = CNTK::NDArrayView::RandomUniform<float>({ 10 }, -0.05, 0.05, 2, device);
    assert(!AreEqual(p2.Value(), v2));

    p2.SetValue(v2);
    if (!AreEqual(p2.Value(), v2))
        throw std::runtime_error("Parameter value does match the expected value.");

    Parameter p3(NDShape({ 3, 4 }), DataType::Float, GlorotUniformInitializer(), device, L"p3");
    auto v3 = CNTK::NDArrayView::RandomUniform<float>({ 3, 4 }, -1, 1, 3, device);

    p3.SetValue(v3);
    if (!AreEqual(p3.Value(), v3))
        throw std::runtime_error("Parameter value does match the expected value.");

    Parameter p4({ 1 }, DataType::Double, Dictionary(), device, L"p4");
    auto v4 = MakeSharedObject<NDArrayView>(1.0, NDShape{ 1 }, device);

    // Since p4 initializer is an empty dictionary, lazy-initialization (triggered by the value getter: p4.Value())
    // should fail. However, the setter will override the bogus initializer and init p4 by copying v4 content.
    p4.SetValue(v4);
    if (!AreEqual(p4.Value(), v4))
        throw std::runtime_error("Parameter value does match the expected value.");
}

void SparseSequenceBatchValueCreationTest(size_t vocabSize, size_t maxAllowedSequenceLength, const DeviceDescriptor& device)
{
    srand(1);
    size_t numSequences = 5;
    auto sequenceLengths = GenerateSequenceLengths(numSequences, maxAllowedSequenceLength);
    std::vector<NDArrayViewPtr> denseSequences(numSequences), sparseSequences(numSequences);
    for (size_t i = 0; i < numSequences; ++i)
        std::tie(denseSequences[i], sparseSequences[i]) = GenerateSparseSequence<float>(vocabSize, sequenceLengths[i], 5);

    // Separately batch the sequences both in dense and sparse forms
    auto denseSequenceBatch = Value::Create({ vocabSize }, denseSequences, device);
    auto sparseSequenceBatch = Value::Create({ vocabSize }, sparseSequences, device);
    auto sparseSequenceBatchDataConvertedToDense = MakeSharedObject<NDArrayView>(DataType::Float, sparseSequenceBatch->Data()->Shape(), device);
    sparseSequenceBatchDataConvertedToDense->CopyFrom(*sparseSequenceBatch->Data());
    auto sparseSequenceBatchValueConvertedToDense = MakeSharedObject<Value>(sparseSequenceBatchDataConvertedToDense, sparseSequenceBatch->Mask());

    if (!Internal::AreEqual(*denseSequenceBatch, *sparseSequenceBatchValueConvertedToDense))
        ReportFailure("Sparse sequence batch does not match expectation");
}

void ValueTests()
{
    fprintf(stderr, "\nValueTests..\n");

    TestSettingParameterValuesManually(DeviceDescriptor::CPUDevice());

    ValueCreationNoNDMaskTest<float>(DeviceDescriptor::CPUDevice(), false);
    ValueCreationNoNDMaskTest<double>(DeviceDescriptor::CPUDevice(), true);
    ValueCreationWithNDMaskTest<double>(DeviceDescriptor::CPUDevice(), false);
    ValueCreationWithNDMaskTest<float>(DeviceDescriptor::CPUDevice(), true);
    ValueCreationOneHotNoNDMaskTest<float>(DeviceDescriptor::CPUDevice(), false);
    ValueCreationOneHotNoNDMaskTest<double>(DeviceDescriptor::CPUDevice(), true);
    ValueCreationOneHotWithNDMaskTest<double>(DeviceDescriptor::CPUDevice(), false);
    ValueCreationOneHotWithNDMaskTest<float>(DeviceDescriptor::CPUDevice(), true);
    SparseSequenceBatchValueCreationTest(300, 7, DeviceDescriptor::CPUDevice());
    SparseSequenceBatchValueCreationTest(2300, 1, DeviceDescriptor::CPUDevice());

    if (IsGPUAvailable())
    {
        TestSettingParameterValuesManually(DeviceDescriptor::GPUDevice(0));

        ValueCreationNoNDMaskTest<double>(DeviceDescriptor::GPUDevice(0), false);
        ValueCreationNoNDMaskTest<float>(DeviceDescriptor::GPUDevice(0), true);
        ValueCreationWithNDMaskTest<float>(DeviceDescriptor::GPUDevice(0), false);
        ValueCreationWithNDMaskTest<double>(DeviceDescriptor::GPUDevice(0), true);
        ValueCreationOneHotNoNDMaskTest<double>(DeviceDescriptor::GPUDevice(0), false);
        ValueCreationOneHotNoNDMaskTest<float>(DeviceDescriptor::GPUDevice(0), true);
        ValueCreationOneHotWithNDMaskTest<float>(DeviceDescriptor::GPUDevice(0), false);
        ValueCreationOneHotWithNDMaskTest<double>(DeviceDescriptor::GPUDevice(0), true);
        SparseSequenceBatchValueCreationTest(50000, 1, DeviceDescriptor::GPUDevice(0));
        SparseSequenceBatchValueCreationTest(6000, 6, DeviceDescriptor::GPUDevice(0));
    }
}
