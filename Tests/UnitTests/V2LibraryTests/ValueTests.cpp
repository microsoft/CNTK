//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <vector>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include "CNTKLibrary.h"
#include "Common.h"

using namespace CNTK;
using namespace std;

namespace CNTK { namespace Test {

void CheckMask(const ValuePtr testValue, const vector<size_t>& seqLenList, const vector<bool>& seqStartFlags)
{
    vector<bool> actualStarts = seqStartFlags;
    if (actualStarts.empty())
    {
        actualStarts.resize(seqLenList.size(), true);
    }
    if (actualStarts.size() != seqLenList.size())
    {
        ReportFailure("The seqStartFlags does not match the number of sequences");
    }
    size_t maxSeqLen = *max_element(seqLenList.begin(), seqLenList.end());

    if (testValue->Mask() == nullptr)
    {
        bool needsMask = (std::find(actualStarts.begin(), actualStarts.end(), false) != actualStarts.end());
        needsMask = needsMask || (std::find_if(seqLenList.begin(), seqLenList.end(), [maxSeqLen](const size_t& currentSequenceLength) {
                                    return (currentSequenceLength != maxSeqLen);
                                 }) != seqLenList.end());
        if (needsMask)
        {
            ReportFailure("A mask for the Value object is expected, but it is not present");
        }
        return;
    }

    auto cpuMask = (testValue->Device().Type() != DeviceKind::CPU) ? testValue->Mask()->DeepClone(DeviceDescriptor::CPUDevice()) : testValue->Mask();
    auto maskData = cpuMask->DataBuffer();

    for (int i = 0; i < seqLenList.size(); i++)
    {
        if ((actualStarts[i] && (maskData[i * maxSeqLen] != MaskKind::SequenceBegin)) || 
            (!actualStarts[i] && (maskData[i * maxSeqLen] != MaskKind::Valid)))
        {
            ReportFailure("The sequence does not have expected value at start.");
        }
        for (size_t j = 1; j < seqLenList[i]; j++)
        {
            if (maskData[i * maxSeqLen + j] != MaskKind::Valid)
            {
                ReportFailure("The sequence should have a Valid mask at position %d", static_cast<int>(i * maxSeqLen + j));
            }
        }
        for (size_t j = seqLenList[i]; j < maxSeqLen; j++)
        {
            if (maskData[i * maxSeqLen + j] != MaskKind::Invalid)
            {
                ReportFailure("The sequence should have a Invalid mask at position %d", static_cast<int>(i * maxSeqLen + j));
            }
        }
    }
}

// Check the actual Value match the expected shape and the given data (in dense format)
template <typename ElementType>
void CheckValue(const ValuePtr testValue, const NDShape& sampleShape, const vector<vector<ElementType>>& expectedData, const vector<size_t>& seqLenList, const vector<bool>& seqStartFlags = {})
{
    size_t sampleSize = sampleShape.TotalSize();
    // Check parameters
    BOOST_TEST(expectedData.size() == seqLenList.size(), "Parameter error: the sequence number in the exepected data and sequence list does not match.");
    for (size_t i = 0; i < expectedData.size(); i++)
    {
        if (expectedData[i].size() != seqLenList[i] * sampleSize)
        {
            ReportFailure("Parameter erroe: the number of data for sequence %" PRIu64 " in the expected data does not match. Expected: %" PRIu64 ", actual: %" PRIu64 ".",
                          i, seqLenList[i] * sampleSize, expectedData[i].size());
        }
    }

    // Check shape 
    auto valueRank = testValue->Shape().Rank();
    auto sampleRank = sampleShape.Rank();
    auto shapeIsCorrect = !((valueRank < sampleRank + 1) || (valueRank > sampleRank + 2) || (sampleShape != testValue->Shape().SubShape(0, sampleRank)));

    BOOST_TEST(shapeIsCorrect, "The Value does not have the expected shape.");

    size_t numOfSequences;
    if (valueRank == sampleShape.Rank() + 1)
    {
        // no batch axis, only sequence axis
        numOfSequences = 1;
    }
    else
    {
        assert(valueRank == sampleShape.Rank() + 2);
        numOfSequences = testValue->Shape()[valueRank - 1];
    }

    if (numOfSequences != expectedData.size())
    {
        ReportFailure("The sequence number in the Value does not match. Expected: %" PRIu64 ", actual: %" PRIu64 ".", expectedData.size(), numOfSequences);
    }

    CheckMask(testValue, seqLenList, seqStartFlags);

    // Get data from Value 
    vector<ElementType> outputData(testValue->Shape().TotalSize());
    NDArrayViewPtr arrayOutput = MakeSharedObject<NDArrayView>(testValue->Shape(), outputData, false);
    arrayOutput->CopyFrom(*testValue->Data());

    size_t maxSeqLen = *max_element(seqLenList.begin(), seqLenList.end());
    size_t oIndex = 0;
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
void CheckValue(const ValuePtr testValue, const size_t dimension, const vector<vector<size_t>>& expectedData, const vector<size_t>& seqLenList, const vector<bool>& seqStartFlags = {})
{
    // Check parameters
    BOOST_TEST(expectedData.size() == seqLenList.size(), "Parameter error: the sequence number in the exepected data and sequence list does not match.");
    for (size_t i = 0; i < expectedData.size(); i++)
    {
        if (expectedData[i].size() != seqLenList[i])
        {
            ReportFailure("Parameter erroe: the number of data for sequence %" PRIu64 " in the expected data does not match. Expected: %" PRIu64 ", actual: %" PRIu64 ".",
                i, seqLenList[i], expectedData[i].size());
        }
    }

    // Check shape
    NDShape shape = testValue->Shape();
    size_t valueRank = shape.Rank();
    if (valueRank < 2 || valueRank > 3 || shape[0] != dimension)
    {
        ReportFailure("The shape of the value does not match\n");
    }
    size_t numOfSequences = valueRank == 2 ? 1 : shape[2]; 
    if (numOfSequences != expectedData.size())
    {
        ReportFailure("The sequence number in the Value does not match. Expected: %" PRIu64 ", actual: %" PRIu64 ".", expectedData.size(), numOfSequences);
    }

    CheckMask(testValue, seqLenList, seqStartFlags);

    // Get data from Value
    vector<ElementType> outputData(shape.TotalSize());
    NDArrayViewPtr arrayOutput = MakeSharedObject<NDArrayView>(shape, outputData, false);
    arrayOutput->CopyFrom(*testValue->Data());

    size_t maxSeqLen = *max_element(seqLenList.begin(), seqLenList.end());
    size_t oIndex = 0;
    for (size_t seq = 0; seq < seqLenList.size(); seq++)
    {
        size_t seqLen = seqLenList[seq];
        for (size_t sample = 0; sample < seqLen; sample++)
        {
            for (size_t c = 0; c < dimension; c++, oIndex++)
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
        oIndex += (maxSeqLen - seqLen) * dimension;
    }
}

template <typename ElementType>
void ValueCreationNoNDMaskTest(const DeviceDescriptor device, bool readOnly)
{
    vector<size_t> dims{3, 2};
    NDShape sampleShape(dims);
    std::vector<std::vector<ElementType>> data;
    ValuePtr testValue;

    // single sequence, single sample
    std::vector<size_t> seqLenList = {1};
    data = GenerateSequences<ElementType>(seqLenList, sampleShape);
    testValue = Value::Create(sampleShape, data, device, readOnly);
    CheckValue(testValue, sampleShape, data, seqLenList);

    // Single sequence, multiple samples
    seqLenList = {2};
    data = GenerateSequences<ElementType>(seqLenList, sampleShape);
    testValue = Value::Create(sampleShape, data, device, readOnly);
    CheckValue(testValue, sampleShape, data, seqLenList);

    // Batch with sequences

    // Same sequence length for testing no NDMask is needed.
    size_t seqLen = 4;
    int testRun = 3;
    size_t maxNumOfSequences = 60;
    // This is only used to generate number of sequnces, so boost distribution is not needed.
    std::default_random_engine generator;
    std::uniform_int_distribution<size_t> distribution(1, maxNumOfSequences);
    for (int i = 0; i < testRun; i++)
    {
        size_t numberOfSequences = distribution(generator);
        std::vector<size_t> seqLenListBatch(numberOfSequences, seqLen);

        data = GenerateSequences<ElementType>(seqLenListBatch, sampleShape);
        // Create the Value object based on the given data and shape.
        testValue = Value::Create(sampleShape, data, device, readOnly);
        // Check whether the created value matches expected shape and data.
        CheckValue(testValue, sampleShape, data, seqLenListBatch);
    }
}

template <typename ElementType>
void ValueCreationWithNDMaskTest(const DeviceDescriptor device, bool readOnly)
{
    vector<size_t> dims{1, 4};
    NDShape sampleShape(dims);
    size_t numberOfSequences; 
    size_t maxAllowedSeqLen = 128;
    std::vector<std::vector<ElementType>> data;
    std::vector<size_t> seqLenList;

    size_t maxNumOfSequences = 80;
    // This is only used to generate number of sequnces, so boost distribution is not needed.
    std::default_random_engine generator;
    std::uniform_int_distribution<size_t> distribution(1, maxNumOfSequences);
    int testRun = 3;
    for (int i = 0; i < testRun; i++)
    {
        numberOfSequences = distribution(generator);
        seqLenList = GenerateSequenceLengths(numberOfSequences, maxAllowedSeqLen);
        data = GenerateSequences<ElementType>(seqLenList, sampleShape);

        ValuePtr testValue = Value::Create(sampleShape, data, device, readOnly);
        CheckValue(testValue, sampleShape, data, seqLenList);
    }

    // Test with only 1 sequence with a sequenceStartFlag=false to ensure a mask
    seqLenList = GenerateSequenceLengths(1, maxAllowedSeqLen);
    data = GenerateSequences<ElementType>(seqLenList, sampleShape);

    ValuePtr testValue = Value::Create(sampleShape, data, {false}, device, readOnly);
    CheckValue(testValue, sampleShape, data, seqLenList, {false} );
}

template <typename ElementType>
void ValueCreationOneHotNoNDMaskTest(const DeviceDescriptor device, bool readOnly)
{
    size_t vocabSize = 18;
    std::vector<std::vector<size_t>> data;
    ValuePtr testValue;

    // Single sequence, single sample
    std::vector<size_t> seqLenList = {1};
    data = GenerateOneHotSequences(seqLenList, vocabSize);
    testValue = Value::Create<ElementType>(vocabSize, data, device, readOnly);
    CheckValue<ElementType>(testValue, vocabSize, data, seqLenList);

    // Single sequence, multiple samples
    seqLenList = {2};
    data = GenerateOneHotSequences(seqLenList, vocabSize);
    testValue = Value::Create<ElementType>(vocabSize, data, device, readOnly);
    CheckValue<ElementType>(testValue, vocabSize, data, seqLenList);

    size_t maxNumOfSequences = 160;
    size_t seqLen = 26;
    // This is only used to generate number of sequnces, so boost distribution is not needed.
    std::default_random_engine generator;
    std::uniform_int_distribution<size_t> distribution(1, maxNumOfSequences);
    int testRun = 3;
    for (int i = 0; i < testRun; i++)
    {
        size_t numberOfSequences = distribution(generator);
        std::vector<size_t> seqLenListBatch(numberOfSequences, seqLen);

        data = GenerateOneHotSequences(seqLenListBatch, vocabSize);
        testValue = Value::Create<ElementType>(vocabSize, data, device, readOnly);
        CheckValue<ElementType>(testValue, vocabSize, data, seqLenListBatch);
    }
}

template <typename ElementType>
void ValueCreationOneHotWithNDMaskTest(const DeviceDescriptor device, bool readOnly)
{
    size_t vocabSize = 64;
    size_t numberOfSequences;
    size_t maxAllowedSeqLen = 95;
    size_t maxSeqLen;
    std::vector<vector<size_t>> data;
    std::vector<size_t> seqLenList;

    size_t maxNumOfSequences = 70;
    // This is only used to generate number of sequnces, so boost distribution is not needed.
    std::default_random_engine generator;
    std::uniform_int_distribution<size_t> distribution(1, maxNumOfSequences);
    int testRun = 3;
    for (int i = 0; i < testRun; i++)
    {
        numberOfSequences = distribution(generator);
        seqLenList = GenerateSequenceLengths(numberOfSequences, maxAllowedSeqLen);
        maxSeqLen = *std::max_element(seqLenList.begin(), seqLenList.end());
        data = GenerateOneHotSequences(seqLenList, vocabSize);
        ValuePtr testValue = Value::Create<ElementType>(vocabSize, data, device, readOnly);
        CheckValue<ElementType>(testValue, vocabSize, data, seqLenList);
    }
}

template <typename ElementType>
void CheckCopyToOutput(const std::vector<std::vector<ElementType>>& expected, const std::vector<std::vector<ElementType>>& actual)
{
    if (expected.size() != actual.size())
        ReportFailure("The number of sequences does not match. expected: %" PRIu64 " actual: %" PRIu64 "\n", expected.size(), actual.size());

    for (size_t i = 0; i < expected.size(); i++)
    {
        if (expected[i].size() != actual[i].size())
        {
            ReportFailure("Seq %lu does not match.\n", static_cast<unsigned long>(i));
        }
        for (size_t j = 0; j < expected[i].size(); j++)
        {
            if (expected[i][j] != actual[i][j])
            {
                ReportFailure("Seq %lu does not match.\n", static_cast<unsigned long>(i));
            }
        }
    }
}

template <typename ElementType>
Variable CreateVariable(NDShape shape, int numOfDynamicAxes)
{
    std::vector<Axis> dynamicAxes;

    switch (numOfDynamicAxes)
    {
    case 0:
        dynamicAxes = {};
        break;
    case 1:
        dynamicAxes = {Axis::DefaultBatchAxis()}; // If only 1 dynamic axis, it is treated as batch axis
        break;
    case 2:
        dynamicAxes = {Axis::DefaultDynamicAxis(), Axis::DefaultBatchAxis()}; // The first is sequence, and the second is batch.
        break;
    default:
        RuntimeError("No more than 2 dynamic axes is allowed.");
    }
    
    Variable sampleVariable(shape, VariableKind::Output, AsDataType<ElementType>(), nullptr, false,
        dynamicAxes, false /*bool isSparse*/, L"sampleVariable", L"sampleVariableUid");

    return sampleVariable;
}

template <typename ElementType>
void TestDenseSequences(const Variable& sampleVariable, std::vector<size_t>& expectedSeqLens, std::vector<std::vector<ElementType>>& output, const DeviceDescriptor& device)
{
    auto input = GenerateSequences<ElementType>(expectedSeqLens, sampleVariable.Shape());
    auto val = Value::Create(sampleVariable.Shape(), input, device);

    val->CopyVariableValueTo(sampleVariable, output);
    CheckCopyToOutput<ElementType>(input, output);
}

template <typename ElementType>
void TestOneHotSequences(const Variable& sampleVariable, std::vector<size_t>& expectedSeqLens, std::vector<std::vector<size_t>>& output, const DeviceDescriptor& device)
{
    auto input = GenerateOneHotSequences(expectedSeqLens, sampleVariable.Shape().TotalSize());
    auto val = Value::Create<ElementType>(sampleVariable.Shape().TotalSize(), input, device);

    val->CopyVariableValueTo(sampleVariable, output);
    CheckCopyToOutput(input, output);
}

template <typename ElementType>
void ValueCopyToDenseTest(const DeviceDescriptor& device)
{
    NDShape sampleShape{{2, 3}};
    std::vector<std::vector<ElementType>> input;
    std::vector<std::vector<ElementType>> output;
    std::vector<size_t> expectedSeqLens;

    //TODO: add tests sparse to dense.

    // Check single sample.
    // No dynamic axis for the sampleVariable
    auto sampleVariable = CreateVariable<ElementType>(sampleShape, 0);
    size_t batchCount = 1;
    expectedSeqLens.clear();
    for (size_t i = 0; i < batchCount; i++)
        expectedSeqLens.push_back(1);
    input = GenerateSequences<ElementType>(expectedSeqLens, sampleShape);
    auto val = Value::Create(sampleShape, input, device);

    val->CopyVariableValueTo(sampleVariable, output);
    CheckCopyToOutput(input, output);

    // 1 dynamic axis (as batch) for the sampleVariable
    sampleVariable = CreateVariable<ElementType>(sampleShape, 1);
    val->CopyVariableValueTo(sampleVariable, output);
    CheckCopyToOutput(input, output);

    // 2 dynamic axes for the sampleVariable
    sampleVariable = CreateVariable<ElementType>(sampleShape, 2);
    val->CopyVariableValueTo(sampleVariable, output);
    CheckCopyToOutput(input, output);

    // Check batch of samples.
    // 1 dynamic axis (as batch) for the sampleVariable
    sampleVariable = CreateVariable<ElementType>(sampleShape, 1);
    batchCount = 2;
    expectedSeqLens.clear();
    for (size_t i = 0; i < batchCount; i++)
        expectedSeqLens.push_back(1);
    input = GenerateSequences<ElementType>(expectedSeqLens, sampleVariable.Shape());
    val = Value::Create(sampleVariable.Shape(), input, device);

    val->CopyVariableValueTo(sampleVariable, output);
    CheckCopyToOutput(input, output);

    // 2 dynamic axes for the sampleVariable
    sampleVariable = CreateVariable<ElementType>(sampleShape, 2);
    val->CopyVariableValueTo(sampleVariable, output);
    CheckCopyToOutput(input, output);

    // Check sequence of samples, but single batch
    // The variable should have 2 dynamic axes.
    sampleVariable = CreateVariable<ElementType>(sampleShape, 2);
    size_t sampleCount = 4;
    batchCount = 1;
    expectedSeqLens.clear();
    for (size_t i = 0; i < batchCount; i++)
        expectedSeqLens.push_back(sampleCount);
    TestDenseSequences(sampleVariable, expectedSeqLens, output, device);

    // Check batch of sequences of the same length, no mask needed.
    batchCount = 4;
    sampleCount = 3;
    expectedSeqLens.clear();
    for (size_t i = 0; i < batchCount; i++)
        expectedSeqLens.push_back(sampleCount);
    TestDenseSequences(sampleVariable, expectedSeqLens, output, device);

    // Check batch of sequecnes with different lengths, mask needed.
    // The length of one sequence is 1.
    std::vector<size_t> sampleCountList = {6, 1, 2};
    batchCount = sampleCountList.size();
    expectedSeqLens.clear();
    for (size_t i = 0; i < batchCount; i++)
        expectedSeqLens.push_back(sampleCountList[i]);
    TestDenseSequences(sampleVariable, expectedSeqLens, output, device);

    // Check batch of sequecnes with different lengths, mask needed.
    sampleCountList = {6, 9, 2};
    batchCount = sampleCountList.size();
    expectedSeqLens.clear();
    for (size_t i = 0; i < batchCount; i++)
        expectedSeqLens.push_back(sampleCountList[i]);
    TestDenseSequences(sampleVariable, expectedSeqLens, output, device);

    // More sequences in a batch, need resize
    sampleCountList = {6, 12, 2, 1, 5, 3, 4};
    batchCount = sampleCountList.size();
    expectedSeqLens.clear();
    for (size_t i = 0; i < batchCount; i++)
        expectedSeqLens.push_back(sampleCountList[i]);
    TestDenseSequences(sampleVariable, expectedSeqLens, output, device);

    // Random batch and sequences
    int testRun = 4;
    size_t maxNumOfSequences = 100;
    size_t maxSequenceLen = 100;
    // This is only used to generate number of sequnces, so boost distribution is not needed.
    std::default_random_engine generator;
    std::uniform_int_distribution<size_t> distribution(1, maxNumOfSequences);
    for (int i = 0; i < testRun; i++)
    {
        batchCount = distribution(generator);

        expectedSeqLens = GenerateSequenceLengths(batchCount, maxSequenceLen);
        input = GenerateSequences<ElementType>(expectedSeqLens, sampleShape);
        val = Value::Create(sampleShape, input, device);

        val->CopyVariableValueTo(sampleVariable, output);
        CheckCopyToOutput( input, output);
    }
}

template <typename ElementType>
void ValueCopyToOneHotTest(const DeviceDescriptor& device)
{
    size_t dim = 100;
    NDShape sampleShape{{dim}};
    std::vector<std::vector<size_t>> input;
    std::vector<std::vector<size_t>> output;
    std::vector<size_t> expectedSeqLens;

    // TODO: add tests dense to sparse
    // Check single sample.
    // No dynamic axis for the sampleVariable.
    auto sampleVariable = CreateVariable<ElementType>(sampleShape, 0);
    size_t batchCount = 1;
    expectedSeqLens.clear();
    for (size_t i = 0; i < batchCount; i++)
        expectedSeqLens.push_back(1);
    input = GenerateOneHotSequences(expectedSeqLens, dim);
    auto val = Value::Create<ElementType>(dim, input, device);

    val->CopyVariableValueTo(sampleVariable, output);
    CheckCopyToOutput(input, output);

    // 1 dynamic axis (as batch) for the sampleVariable
    sampleVariable = CreateVariable<ElementType>(sampleShape, 1);
    val->CopyVariableValueTo(sampleVariable, output);
    CheckCopyToOutput(input, output);

    // 2 dynamic axes for the sampleVariable
    sampleVariable = CreateVariable<ElementType>(sampleShape, 2);
    val->CopyVariableValueTo(sampleVariable, output);
    CheckCopyToOutput(input, output);

    // Check batch of samples.
    // 1 dynamic axis (as batch) for the sampleVariable
    sampleVariable = CreateVariable<ElementType>(sampleShape, 1);
    batchCount = 2;
    expectedSeqLens.clear();
    for (size_t i = 0; i < batchCount; i++)
        expectedSeqLens.push_back(1);
    input = GenerateOneHotSequences(expectedSeqLens, sampleVariable.Shape().TotalSize());
    val = Value::Create<ElementType>(sampleVariable.Shape().TotalSize(), input, device);

    val->CopyVariableValueTo(sampleVariable, output);
    CheckCopyToOutput(input, output);

    // 2 dynamic axes for the sampleVariable
    sampleVariable = CreateVariable<ElementType>(sampleShape, 2);
    val->CopyVariableValueTo(sampleVariable, output);
    CheckCopyToOutput(input, output);

    // Check sequence of samples, but single batch
    // The variable should have 2 dynamic axes.
    sampleVariable = CreateVariable<ElementType>(sampleShape, 2);
    size_t sampleCount = 4;
    batchCount = 1;
    expectedSeqLens.clear();
    for (size_t i = 0; i < batchCount; i++)
        expectedSeqLens.push_back(sampleCount);
    TestOneHotSequences<ElementType>(sampleVariable, expectedSeqLens, output, device);

    // Check batch of sequences of the same length, no mask needed.
    batchCount = 4;
    sampleCount = 3;
    expectedSeqLens.clear();
    for (size_t i = 0; i < batchCount; i++)
        expectedSeqLens.push_back(sampleCount);
    TestOneHotSequences<ElementType>(sampleVariable, expectedSeqLens, output, device);

    // Check batch of sequecnes with different lengths, mask needed.
    // The length of one sequence is 1.
    std::vector<size_t> sampleCountList = {6, 1, 2};
    batchCount = sampleCountList.size();
    expectedSeqLens.clear();
    for (size_t i = 0; i < batchCount; i++)
        expectedSeqLens.push_back(sampleCountList[i]);
    TestOneHotSequences<ElementType>(sampleVariable, expectedSeqLens, output, device);

    // Check batch of sequecnes with different lengths, mask needed.
    sampleCountList = {6, 9, 2};
    batchCount = sampleCountList.size();
    expectedSeqLens.clear();
    for (size_t i = 0; i < batchCount; i++)
        expectedSeqLens.push_back(sampleCountList[i]);
    TestOneHotSequences<ElementType>(sampleVariable, expectedSeqLens, output, device);

    // More sequences in a batch, resize required
    sampleCountList = {6, 12, 2, 1, 5, 3, 4};
    batchCount = sampleCountList.size();
    expectedSeqLens.clear();
    for (size_t i = 0; i < batchCount; i++)
        expectedSeqLens.push_back(sampleCountList[i]);
    TestOneHotSequences<ElementType>(sampleVariable, expectedSeqLens, output, device);

    // Random batch and sequences
    int testRun = 4;
    size_t maxNumOfSequences = 100;
    size_t maxSequenceLen = 100;
    // This is only used to generate number of sequnces, so boost distribution is not needed.
    std::default_random_engine generator;
    std::uniform_int_distribution<size_t> distribution(1, maxNumOfSequences);
    for (int i = 0; i < testRun; i++)
    {
        batchCount = distribution(generator);

        expectedSeqLens = GenerateSequenceLengths(batchCount, maxSequenceLen);
        input = GenerateOneHotSequences(expectedSeqLens, dim);
        val = Value::Create<ElementType>(dim, input, device);

        val->CopyVariableValueTo(sampleVariable, output);
        CheckCopyToOutput(input, output);
    }
}

void ValueCopyToExceptionsTest(const DeviceDescriptor& device)
{
    std::vector<size_t> expectedSeqLens = {1};
    std::vector<std::vector<float>> input;
    std::vector<std::vector<float>> output;
    std::vector<std::vector<double>> outputInDouble;
    std::vector<std::vector<size_t>> outputInOneHot;
    NDShape sampleShape{{2, 3}};
    NDShape sampleOneHotShape{{100}};

    input = GenerateSequences<float>(expectedSeqLens, sampleShape);
    auto val = Value::Create(sampleShape, input, device);

    // Test variable with unknown shape
    auto sampleVariable = CreateVariable<float>(NDShape::Unknown, 0);
    VerifyException([&val, &sampleVariable, &output]() {
        val->CopyVariableValueTo(sampleVariable, output);
    }, "The expected exception has not been caught: It is not supported that the outputVariable has a unknown shape or inferred dimension.");

    // Test variable having shape with InferredDimentsion.
    sampleVariable = CreateVariable<float>(NDShape(2), 0);
    VerifyException([&val, &sampleVariable, &output]() {
        val->CopyVariableValueTo(sampleVariable, output);
    }, "The expected exception has not been caught: It is not supported that the outputVariable has a unknown shape or inferred dimension.");

    // Test variable having incorrect data type.
    sampleVariable = CreateVariable<double>(sampleShape, 0);
    VerifyException([&val, &sampleVariable, &output]() {
        val->CopyVariableValueTo(sampleVariable, output);
    }, "The expected exception has not been caught: The outputVariable has a different data type than the Value object.");

    sampleVariable = CreateVariable<double>(sampleOneHotShape, 0);
    VerifyException([&val, &sampleVariable, &outputInOneHot]() {
        val->CopyVariableValueTo(sampleVariable, outputInOneHot);
    }, "The expected exception has not been caught: The outputVariable has a different data type than the Value object.");

    // Test output buffer having incorrect data type.
    sampleVariable = CreateVariable<float>(sampleShape, 0);
    VerifyException([&val, &sampleVariable, &outputInDouble]() {
        val->CopyVariableValueTo(sampleVariable, outputInDouble);
    }, "The expected exception has not been caught: The specified ElementType Double does not match the DataType Float");

    // Test the first axis when using one-hot format.
    VerifyException([&val, &sampleVariable, &outputInOneHot]() {
        val->CopyVariableValueTo(sampleVariable, outputInOneHot);
    }, "The expected exception has not been caught: The outputVariable's leading axis dimensionality must equal the total size of the variable for sparse data.");
}


void TestSettingParameterValuesManually(const DeviceDescriptor& device)
{
    auto v1_1 = MakeSharedObject<NDArrayView>(0.5, NDShape({ 2, 2 }), device);
    auto v1_2 = MakeSharedObject<NDArrayView>(0.4, NDShape({ 2, 2 }), device);

    Parameter p1(v1_1);
    auto value = p1.Value();

    assert(!AreEqual(v1_1, v1_2) && !AreEqual(p1.Value(), v1_2));

    p1.SetValue(v1_2);
    BOOST_TEST(AreEqual(p1.Value(), v1_2), "Parameter value does match the expected value.");

    Parameter p2(NDArrayView::RandomUniform<float>({ 10 }, -0.05, 0.05, 1, device));
    auto v2 = NDArrayView::RandomUniform<float>({ 10 }, -0.05, 0.05, 2, device);
    assert(!AreEqual(p2.Value(), v2));

    p2.SetValue(v2);
    BOOST_TEST(AreEqual(p2.Value(), v2), "Parameter value does match the expected value.");

    Parameter p3(NDShape({ 3, 4 }), DataType::Float, GlorotUniformInitializer(), device, L"p3");
    auto v3 = NDArrayView::RandomUniform<float>({ 3, 4 }, -1, 1, 3, device);

    p3.SetValue(v3);
    BOOST_TEST(AreEqual(p3.Value(), v3), "Parameter value does match the expected value.");

    Parameter p4({ 1 }, DataType::Double, Dictionary(), device, L"p4");
    auto v4 = MakeSharedObject<NDArrayView>(1.0, NDShape{ 1 }, device);

    // Since p4 initializer is an empty dictionary, lazy-initialization (triggered by the value getter: p4.Value())
    // should fail. However, the setter will override the bogus initializer and init p4 by copying v4 content.
    p4.SetValue(v4);
    BOOST_TEST(AreEqual(p4.Value(), v4), "Parameter value does match the expected value.");
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

    BOOST_TEST(Internal::AreEqual(*denseSequenceBatch, *sparseSequenceBatchValueConvertedToDense), "Sparse sequence batch does not match expectation");
}

template <typename ElementType>
void CreateBatchTestDense(const DeviceDescriptor device, bool readOnly)
{
    size_t numAxes = 3;
    size_t maxDimSize = 20; 
    NDShape sampleShape = CreateShape(numAxes, maxDimSize);
    auto sampleSize = sampleShape.TotalSize();

    size_t batchCount = 1;
    size_t maxSequenceLen = 100;
    auto seqLenList = GenerateSequenceLengths(batchCount, maxSequenceLen);
    // Here we miss use GenertaeSequences to create batch.
    auto data = GenerateSequences<ElementType>(seqLenList, sampleShape);
    auto batch2 = data[0];
    auto testValue = Value::CreateBatch(sampleShape, batch2, device, readOnly);
    vector<vector<ElementType>> expectedResult;
    for (size_t i = 0; i < data[0].size(); i += sampleSize)
    {
        expectedResult.push_back(vector<ElementType>(data[0].begin() + i, data[0].begin() + i + sampleSize));
    }
    vector<size_t> resultSeqLen(data[0].size()/sampleSize, 1);
    CheckValue(testValue, sampleShape, expectedResult, resultSeqLen);

    vector<ElementType> wrongBatch(sampleSize * 2 - 1, 0);
    VerifyException([&sampleShape, &wrongBatch, &device, &readOnly]() {
        Value::CreateBatch(sampleShape, wrongBatch, device, readOnly);
    }, "The expected exception has not been caught: The number of data is not a multiple of the sample size.");

    auto emptyBatch = vector<ElementType>(0);
    VerifyException([&sampleShape, &emptyBatch, &device, &readOnly]() {
        Value::CreateBatch(sampleShape, emptyBatch, device, readOnly);
    }, "The expected exception has not been caught: The number of sequences is 0");
}

template <typename ElementType>
void CreateSequenceTestDense(const DeviceDescriptor device, bool readOnly)
{
    size_t numAxes = 4;
    size_t maxDimSize = 30;
    NDShape sampleShape = CreateShape(numAxes, maxDimSize);
    auto sampleSize = sampleShape.TotalSize();

    size_t batchCount = 1;
    size_t maxSequenceLen = 60;

    // Test without using seqStartFlag
    auto seqLenList = GenerateSequenceLengths(batchCount, maxSequenceLen);
    auto data = GenerateSequences<ElementType>(seqLenList, sampleShape);
    auto seq = data[0];
    auto testValue = Value::CreateSequence(sampleShape, seq, device, readOnly);
    CheckValue(testValue, sampleShape, data, seqLenList);

    // Test seqStartFlag is true
    seqLenList = GenerateSequenceLengths(batchCount, maxSequenceLen);
    data = GenerateSequences<ElementType>(seqLenList, sampleShape);
    seq = data[0];
    testValue = Value::CreateSequence(sampleShape, seq, true, device, readOnly);
    CheckValue(testValue, sampleShape, data, seqLenList, { true });

    // Test seqStartFlag is false
    seqLenList = GenerateSequenceLengths(batchCount, maxSequenceLen);
    data = GenerateSequences<ElementType>(seqLenList, sampleShape);
    seq = data[0];
    testValue = Value::CreateSequence(sampleShape, seq, false, device, readOnly);
    CheckValue(testValue, sampleShape, data, seqLenList, { false });

    vector<ElementType> wrongSeq(sampleSize * 2 - 1, 0);
    VerifyException([&sampleShape, &wrongSeq, &device, &readOnly]() {
        Value::CreateSequence(sampleShape, wrongSeq, device, readOnly);
    }, "The expected exception has not been caught: The number of data is not a multiple of the sample size.");

    auto emptySeq = vector<ElementType>(0);
    VerifyException([&sampleShape, &emptySeq, &device, &readOnly]() {
        Value::CreateSequence(sampleShape, emptySeq, device, readOnly);
    }, "The expected exception has not been caught: The sequence length is 0");
}


template <typename ElementType>
void CreateBatchOfSequencesTestDense(const DeviceDescriptor device, bool readOnly)
{
    size_t numAxes = 3;
    size_t maxDimSize = 10;
    NDShape sampleShape = CreateShape(numAxes, maxDimSize);
    size_t maxNumOfSequences = 20;
    size_t maxAllowedSequenceLen = 10;
    size_t batchCount;

    // Explicitly test seqStartFlags
    batchCount = 2;
    auto seqLenList = GenerateSequenceLengths(batchCount, maxAllowedSequenceLen);
    auto data = GenerateSequences<ElementType>(seqLenList, sampleShape);
    vector<bool> seqStartFlags = { true, true };
    auto testValue = Value::CreateBatchOfSequences(sampleShape, data, seqStartFlags, device, readOnly);
    CheckValue(testValue, sampleShape, data, seqLenList, seqStartFlags);

    seqLenList = GenerateSequenceLengths(batchCount, maxAllowedSequenceLen);
    data = GenerateSequences<ElementType>(seqLenList, sampleShape);
    seqStartFlags = { false, false };
    testValue = Value::CreateBatchOfSequences(sampleShape, data, seqStartFlags, device, readOnly);
    CheckValue(testValue, sampleShape, data, seqLenList, seqStartFlags);

    batchCount = 3;
    seqLenList = GenerateSequenceLengths(batchCount, maxAllowedSequenceLen);
    data = GenerateSequences<ElementType>(seqLenList, sampleShape);
    seqStartFlags = { true, false, true };
    testValue = Value::CreateBatchOfSequences(sampleShape, data, seqStartFlags, device, readOnly);
    CheckValue(testValue, sampleShape, data, seqLenList, seqStartFlags);

    int testRun = 4;
    std::default_random_engine generator;
    std::uniform_int_distribution<size_t> distribution(1, maxNumOfSequences);
    for (int i = 0; i < testRun; i++)
    {
        batchCount = distribution(generator);
        seqLenList = GenerateSequenceLengths(batchCount, maxAllowedSequenceLen);
        data = GenerateSequences<ElementType>(seqLenList, sampleShape);
        testValue = Value::CreateBatchOfSequences(sampleShape, data, device, readOnly);
        CheckValue(testValue, sampleShape, data, seqLenList);

        seqLenList = GenerateSequenceLengths(batchCount, maxAllowedSequenceLen);
        data = GenerateSequences<ElementType>(seqLenList, sampleShape);
        seqStartFlags = GenerateSequenceStartFlags(batchCount);
        testValue = Value::CreateBatchOfSequences(sampleShape, data, seqStartFlags, device, readOnly);
        CheckValue(testValue, sampleShape, data, seqLenList, seqStartFlags);
    }
}


template <typename ElementType>
void CreateBatchTestOneHot(const DeviceDescriptor device, bool readOnly)
{
    size_t maxDimSize = 30;
    std::default_random_engine dimSizeGenerator;
    std::uniform_int_distribution<size_t> dimSizeDistribution(1, maxDimSize);
    size_t dimSize = dimSizeDistribution(dimSizeGenerator);
    size_t batchCount = 1;
    size_t maxSequenceLen = 100;

    auto seqLenList = GenerateSequenceLengths(batchCount, maxSequenceLen);
    // Here we miss use GenertaeSequences to create batch.
    auto data = GenerateOneHotSequences(seqLenList, dimSize);
    auto batch2 = data[0];
    auto testValue = Value::CreateBatch<ElementType>(dimSize, batch2, device, readOnly);
    vector<vector<size_t>> expectedResult;
    for (size_t i = 0; i < data[0].size(); i ++)
    {
        expectedResult.push_back(vector<size_t>(1, data[0][i]));
    }
    vector<size_t> resultSeqLen(data[0].size(), 1);
    CheckValue<ElementType>(testValue, dimSize, expectedResult, resultSeqLen);

    auto emptyBatch = vector<size_t>(0);
    VerifyException([&dimSize, &emptyBatch, &device, &readOnly]() {
        Value::CreateBatch<ElementType>(dimSize, emptyBatch, device, readOnly);
    }, "The expected exception has not been caught: The number of sequences is 0");
}

template <typename ElementType>
void CreateSequenceTestOneHot(const DeviceDescriptor device, bool readOnly)
{
    size_t maxDimSize = 210;
    std::default_random_engine dimSizeGenerator;
    std::uniform_int_distribution<size_t> dimSizeDistribution(1, maxDimSize);
    size_t dimSize = dimSizeDistribution(dimSizeGenerator);
    size_t batchCount = 1;
    size_t maxSequenceLen = 40;

    // Test without using seqStartFlag
    auto seqLenList = GenerateSequenceLengths(batchCount, maxSequenceLen);
    auto data = GenerateOneHotSequences(seqLenList, dimSize);
    auto seq = data[0];
    auto testValue = Value::CreateSequence<ElementType>(dimSize, seq, device, readOnly);
    CheckValue<ElementType>(testValue, dimSize, data, seqLenList);

    // Test seqStartFlag is true
    dimSize = dimSizeDistribution(dimSizeGenerator);
    seqLenList = GenerateSequenceLengths(batchCount, maxSequenceLen);
    data = GenerateOneHotSequences(seqLenList, dimSize);
    seq = data[0];
    testValue = Value::CreateSequence<ElementType>(dimSize, seq, true, device, readOnly);
    CheckValue<ElementType>(testValue, dimSize, data, seqLenList, { true });

    // Test seqStartFlag is false
    dimSize = dimSizeDistribution(dimSizeGenerator);
    seqLenList = GenerateSequenceLengths(batchCount, maxSequenceLen);
    data = GenerateOneHotSequences(seqLenList, dimSize);
    seq = data[0];
    testValue = Value::CreateSequence<ElementType>(dimSize, seq, false, device, readOnly);
    CheckValue<ElementType>(testValue, dimSize, data, seqLenList, { false });

    auto emptySeq = vector<size_t>(0);
    VerifyException([&dimSize, &emptySeq, &device, &readOnly]() {
        Value::CreateSequence<ElementType>(dimSize, emptySeq, device, readOnly);
    }, "The expected exception has not been caught: The sequences length is 0");
}


template <typename ElementType>
void CreateBatchOfSequencesTestOneHot(const DeviceDescriptor device, bool readOnly)
{
    size_t maxDimSize = 40;
    std::default_random_engine dimSizeGenerator;
    std::uniform_int_distribution<size_t> dimSizeDistribution(1, maxDimSize);
    size_t dimSize = dimSizeDistribution(dimSizeGenerator);
    size_t maxNumOfSequences = 20;
    size_t maxAllowedSequenceLen = 10;
    size_t batchCount;

    // Explicitly test seqStartFlags
    batchCount = 2;
    dimSize = dimSizeDistribution(dimSizeGenerator);
    auto seqLenList = GenerateSequenceLengths(batchCount, maxAllowedSequenceLen);
    auto data = GenerateOneHotSequences(seqLenList, dimSize);
    vector<bool> seqStartFlags = { true, true };
    auto testValue = Value::CreateBatchOfSequences<ElementType>(dimSize, data, seqStartFlags, device, readOnly);
    CheckValue<ElementType>(testValue, dimSize, data, seqLenList, seqStartFlags);

    dimSize = dimSizeDistribution(dimSizeGenerator);
    seqLenList = GenerateSequenceLengths(batchCount, maxAllowedSequenceLen);
    data = GenerateOneHotSequences(seqLenList, dimSize);
    seqStartFlags = { false, false };
    testValue = Value::CreateBatchOfSequences<ElementType>(dimSize, data, seqStartFlags, device, readOnly);
    CheckValue<ElementType>(testValue, dimSize, data, seqLenList, seqStartFlags);

    batchCount = 3;
    dimSize = dimSizeDistribution(dimSizeGenerator);
    seqLenList = GenerateSequenceLengths(batchCount, maxAllowedSequenceLen);
    data = GenerateOneHotSequences(seqLenList, dimSize);
    seqStartFlags = { true, false, true };
    testValue = Value::CreateBatchOfSequences<ElementType>(dimSize, data, seqStartFlags, device, readOnly);
    CheckValue<ElementType>(testValue, dimSize, data, seqLenList, seqStartFlags);

    int testRun = 4;
    std::default_random_engine generator;
    std::uniform_int_distribution<size_t> distribution(1, maxNumOfSequences);
    for (int i = 0; i < testRun; i++)
    {
        batchCount = distribution(generator);
        dimSize = dimSizeDistribution(dimSizeGenerator);
        seqLenList = GenerateSequenceLengths(batchCount, maxAllowedSequenceLen);
        data = GenerateOneHotSequences(seqLenList, dimSize);
        testValue = Value::CreateBatchOfSequences<ElementType>(dimSize, data, device, readOnly);
        CheckValue<ElementType>(testValue, dimSize, data, seqLenList);

        seqLenList = GenerateSequenceLengths(batchCount, maxAllowedSequenceLen);
        data = GenerateOneHotSequences(seqLenList, dimSize);
        seqStartFlags = GenerateSequenceStartFlags(batchCount);
        testValue = Value::CreateBatchOfSequences<ElementType>(dimSize, data, seqStartFlags, device, readOnly);
        CheckValue<ElementType>(testValue, dimSize, data, seqLenList, seqStartFlags);
    }
}

void CheckSparseValueEqualToDenseValue(ValuePtr sparseValue, ValuePtr denseValue, const DeviceDescriptor device)
{
    auto sparseValueInDenseView = MakeSharedObject<NDArrayView>(sparseValue->GetDataType(), sparseValue->Shape(), device);
    sparseValueInDenseView->CopyFrom(*sparseValue->Data());
    auto sparseValueInDense = MakeSharedObject<Value>(sparseValueInDenseView, sparseValue->Mask());
    BOOST_TEST(Internal::AreEqual(*denseValue, *sparseValueInDense), "Value created using sparse input does not match expectation");
}

template <typename ElementType>
void CreateSequenceTestSprse(const DeviceDescriptor device, bool readOnly)
{
    size_t maxDimSize = 110;
    std::default_random_engine dimSizeGenerator;
    std::uniform_int_distribution<size_t> dimSizeDistribution(1, maxDimSize);
    size_t maxSequenceLen = 50;

    std::vector<ElementType> referenceDenseData;
    std::vector<SparseIndexType> colsStarts;
    std::vector<SparseIndexType> rowIndices;
    std::vector<ElementType> nonZeroValues;
    size_t numNonZeroValues;
    size_t dimSize;
    size_t seqLen;
    ValuePtr sparseValue, denseValue;

    int testRun = 4;
    for (int i = 0; i < testRun; i++)
    {
        // Test without using seqStartFlag
        dimSize = dimSizeDistribution(dimSizeGenerator);
        seqLen = GenerateSequenceLengths(1, maxSequenceLen)[0];
        std::tie(referenceDenseData, colsStarts, rowIndices, nonZeroValues, numNonZeroValues) = GenerateSequenceInCSC<ElementType>(dimSize, seqLen);

        // Not using seqStartFlag.
        sparseValue = Value::CreateSequence<ElementType>( { dimSize }, seqLen, colsStarts.data(), rowIndices.data(), nonZeroValues.data(), numNonZeroValues, device, readOnly);
        denseValue = Value::CreateSequence<ElementType>({ dimSize }, referenceDenseData, device, readOnly);
        CheckSparseValueEqualToDenseValue(sparseValue, denseValue, device);

        // Using seqStartFlag.
        auto seqStartFlag = static_cast<int>(rand()) % 2 == 0 ? true : false;
        sparseValue = Value::CreateSequence({ dimSize }, seqLen, colsStarts.data(), rowIndices.data(), nonZeroValues.data(), numNonZeroValues, seqStartFlag, device, readOnly);
        denseValue = Value::CreateSequence({ dimSize }, referenceDenseData, seqStartFlag, device, readOnly);
        CheckSparseValueEqualToDenseValue(sparseValue, denseValue, device);
    }

    dimSize = dimSizeDistribution(dimSizeGenerator);
    seqLen = GenerateSequenceLengths(1, maxSequenceLen)[0];
    std::tie(referenceDenseData, colsStarts, rowIndices, nonZeroValues, numNonZeroValues) = GenerateSequenceInCSC<ElementType>(dimSize * dimSize, seqLen);

    sparseValue = Value::CreateSequence({ dimSize * dimSize }, seqLen, colsStarts.data(), rowIndices.data(), nonZeroValues.data(), numNonZeroValues, device, readOnly);
    denseValue = Value::CreateSequence({ dimSize * dimSize }, referenceDenseData, device, readOnly);
    CheckSparseValueEqualToDenseValue(sparseValue, denseValue, device);

    VerifyException([&colsStarts, &rowIndices, &nonZeroValues, &numNonZeroValues, &dimSize, &seqLen, &device, &readOnly]() {
        Value::CreateSequence<ElementType>( {dimSize, dimSize }, seqLen, colsStarts.data(), rowIndices.data(), nonZeroValues.data(), numNonZeroValues, device, readOnly);
    }, "The expected exception has not been caught: Value::Create: the sample shape leading axis dimensionality must equal the total size of the sample.");
}

struct ValueFixture
{
    ValueFixture()
    {
        srand(1);
    }
};

BOOST_FIXTURE_TEST_SUITE(ValueSuite, ValueFixture)

BOOST_AUTO_TEST_CASE(SettingParameterValuesManuallyInCPU)
{
    if (!ShouldRunOnCpu())
        return;

    TestSettingParameterValuesManually(DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_CASE(ValueCreationWithoutNDMaskInCPU)
{
    if (!ShouldRunOnCpu())
        return;

    ValueCreationNoNDMaskTest<float>(DeviceDescriptor::CPUDevice(), false);
    ValueCreationNoNDMaskTest<double>(DeviceDescriptor::CPUDevice(), true);
}

BOOST_AUTO_TEST_CASE(ValueCreationWithNDMaskInCPU)
{
    if (!ShouldRunOnCpu())
        return;

    ValueCreationWithNDMaskTest<double>(DeviceDescriptor::CPUDevice(), false);
    ValueCreationWithNDMaskTest<float>(DeviceDescriptor::CPUDevice(), true);
}

BOOST_AUTO_TEST_CASE(ValueCreationOneHotWithoutNDMaskInCPU)
{
    if (!ShouldRunOnCpu())
        return;

    ValueCreationOneHotNoNDMaskTest<float>(DeviceDescriptor::CPUDevice(), false);
    ValueCreationOneHotNoNDMaskTest<double>(DeviceDescriptor::CPUDevice(), true);
}

BOOST_AUTO_TEST_CASE(ValueCreationOneHotWithNDMaskInCPU)
{
    if (!ShouldRunOnCpu())
        return;

    ValueCreationOneHotWithNDMaskTest<double>(DeviceDescriptor::CPUDevice(), false);
    ValueCreationOneHotWithNDMaskTest<float>(DeviceDescriptor::CPUDevice(), true);
}

BOOST_AUTO_TEST_CASE(SparseSequenceBatchValueCreationInCPU)
{
    if (!ShouldRunOnCpu())
        return;

    SparseSequenceBatchValueCreationTest(300, 7, DeviceDescriptor::CPUDevice());
    SparseSequenceBatchValueCreationTest(2300, 1, DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_CASE(ValueCopyToDenseInCPU)
{
    if (!ShouldRunOnCpu())
        return;

    ValueCopyToDenseTest<float>(DeviceDescriptor::CPUDevice());
    ValueCopyToDenseTest<double>(DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_CASE(ValueCopyToOneHotTestInCPU)
{
    if (!ShouldRunOnCpu())
        return;

    ValueCopyToOneHotTest<float>(DeviceDescriptor::CPUDevice());
    ValueCopyToOneHotTest<double>(DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_CASE(ValueCopyToExceptionsInCPU)
{
    if (!ShouldRunOnCpu())
        return;

    ValueCopyToExceptionsTest(DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_CASE(CreateBatchDenseInCPU)
{
    if (!ShouldRunOnCpu())
        return;

    CreateBatchTestDense<float>(DeviceDescriptor::CPUDevice(), true);
    CreateBatchTestDense<double>(DeviceDescriptor::CPUDevice(), false);
}

BOOST_AUTO_TEST_CASE(CreateSequenceDenseInCPU)
{
    if (!ShouldRunOnCpu())
        return;

    CreateSequenceTestDense<float>(DeviceDescriptor::CPUDevice(), true);
    CreateSequenceTestDense<double>(DeviceDescriptor::CPUDevice(), false);
}

BOOST_AUTO_TEST_CASE(CreateBatchOfSequencesDenseInCPU)
{
    if (!ShouldRunOnCpu())
        return;

    CreateBatchOfSequencesTestDense<float>(DeviceDescriptor::CPUDevice(), true);
    CreateBatchOfSequencesTestDense<double>(DeviceDescriptor::CPUDevice(), false);
}

BOOST_AUTO_TEST_CASE(CreateBatchOneHotInCPU)
{
    if (!ShouldRunOnCpu())
        return;

    CreateBatchTestOneHot<float>(DeviceDescriptor::CPUDevice(), true);
    CreateBatchTestOneHot<double>(DeviceDescriptor::CPUDevice(), false);
}

BOOST_AUTO_TEST_CASE(CreateSequenceOneHotInCPU)
{
    if (!ShouldRunOnCpu())
        return;

    CreateSequenceTestOneHot<float>(DeviceDescriptor::CPUDevice(), true);
    CreateSequenceTestOneHot<double>(DeviceDescriptor::CPUDevice(), false);
}

BOOST_AUTO_TEST_CASE(CreateBatchOfSequencesOneHotInCPU)
{
    if (!ShouldRunOnCpu())
        return;

    CreateBatchOfSequencesTestOneHot<float>(DeviceDescriptor::CPUDevice(), true);
    CreateBatchOfSequencesTestOneHot<double>(DeviceDescriptor::CPUDevice(), false);
}

BOOST_AUTO_TEST_CASE(CreateSequenceSparseInCPU)
{
    if (!ShouldRunOnCpu())
        return;

    CreateSequenceTestSprse<float>(DeviceDescriptor::CPUDevice(), false);
    CreateSequenceTestSprse<double>(DeviceDescriptor::CPUDevice(), true);
}

BOOST_AUTO_TEST_CASE(SettingParameterValuesManuallyInGPU)
{
    if (ShouldRunOnGpu())
        TestSettingParameterValuesManually(DeviceDescriptor::GPUDevice(0));
}

BOOST_AUTO_TEST_CASE(ValueCreationWithNDMaskInGPU)
{
    if (ShouldRunOnGpu())
    {
        ValueCreationWithNDMaskTest<float>(DeviceDescriptor::GPUDevice(0), false);
        ValueCreationWithNDMaskTest<double>(DeviceDescriptor::GPUDevice(0), true);
    }
}

BOOST_AUTO_TEST_CASE(ValueCreationWithoutNDMaskInGPU)
{
    if (ShouldRunOnGpu())
    {
        ValueCreationNoNDMaskTest<double>(DeviceDescriptor::GPUDevice(0), false);
        ValueCreationNoNDMaskTest<float>(DeviceDescriptor::GPUDevice(0), true);
    }

}

BOOST_AUTO_TEST_CASE(ValueCreationOneHotWithoutNDMaskInGPU)
{
    if (ShouldRunOnGpu())
    {
        ValueCreationOneHotNoNDMaskTest<double>(DeviceDescriptor::GPUDevice(0), false);
        ValueCreationOneHotNoNDMaskTest<float>(DeviceDescriptor::GPUDevice(0), true);
    }

}

BOOST_AUTO_TEST_CASE(ValueCreationOneHotWithNDMaskInGPU)
{
    if (ShouldRunOnGpu())
    {
        ValueCreationOneHotWithNDMaskTest<float>(DeviceDescriptor::GPUDevice(0), false);
        ValueCreationOneHotWithNDMaskTest<double>(DeviceDescriptor::GPUDevice(0), true);
    }
}

BOOST_AUTO_TEST_CASE(SparseSequenceBatchValueCreationInGPU)
{
    if (ShouldRunOnGpu())
    {
        SparseSequenceBatchValueCreationTest(50000, 1, DeviceDescriptor::GPUDevice(0));
        SparseSequenceBatchValueCreationTest(6000, 6, DeviceDescriptor::GPUDevice(0));
    }
}

BOOST_AUTO_TEST_CASE(ValueCopyToDenseInGPU)
{
    if (ShouldRunOnGpu())
    {
        ValueCopyToDenseTest<float>(DeviceDescriptor::GPUDevice(0));
        ValueCopyToDenseTest<double>(DeviceDescriptor::GPUDevice(0));
    }
}

BOOST_AUTO_TEST_CASE(ValueCopyToOneHotInGPU)
{
    if (ShouldRunOnGpu())
    {
        ValueCopyToOneHotTest<float>(DeviceDescriptor::GPUDevice(0));
        ValueCopyToOneHotTest<double>(DeviceDescriptor::GPUDevice(0));
    }
}

BOOST_AUTO_TEST_CASE(ValueCopyToExceptionsInGPU)
{
    if (ShouldRunOnGpu())
    {
        ValueCopyToExceptionsTest(DeviceDescriptor::GPUDevice(0));
    }
}

BOOST_AUTO_TEST_CASE(CreateBatchDenseInGPU)
{
    if (ShouldRunOnGpu())
    {
        CreateBatchTestDense<float>(DeviceDescriptor::GPUDevice(0), false);
        CreateBatchTestDense<double>(DeviceDescriptor::GPUDevice(0), true);
    }
}

BOOST_AUTO_TEST_CASE(CreateSequenceDenseInGPU)
{
    if (ShouldRunOnGpu())
    {
        CreateSequenceTestDense<float>(DeviceDescriptor::GPUDevice(0), false);
        CreateSequenceTestDense<double>(DeviceDescriptor::GPUDevice(0), true);
    }
}

BOOST_AUTO_TEST_CASE(CreateBatchOfSequencesDenseInGPU)
{
    if (ShouldRunOnGpu())
    {
        CreateBatchOfSequencesTestDense<float>(DeviceDescriptor::GPUDevice(0), false);
        CreateBatchOfSequencesTestDense<double>(DeviceDescriptor::GPUDevice(0), true);
    }
}

BOOST_AUTO_TEST_CASE(CreateBatchOneHotInGPU)
{
    if (ShouldRunOnGpu())
    {
        CreateBatchTestOneHot<float>(DeviceDescriptor::GPUDevice(0), false);
        CreateBatchTestOneHot<double>(DeviceDescriptor::GPUDevice(0), true);
    }
}

BOOST_AUTO_TEST_CASE(CreateSequenceOneHotInGPU)
{
    if (ShouldRunOnGpu())
    {
        CreateSequenceTestOneHot<float>(DeviceDescriptor::GPUDevice(0), false);
        CreateSequenceTestOneHot<double>(DeviceDescriptor::GPUDevice(0), true);
    }
}

BOOST_AUTO_TEST_CASE(CreateBatchOfSequencesOneHotInGPU)
{
    if (ShouldRunOnGpu())
    {
        CreateBatchOfSequencesTestOneHot<float>(DeviceDescriptor::GPUDevice(0), false);
        CreateBatchOfSequencesTestOneHot<double>(DeviceDescriptor::GPUDevice(0), true);
    }
}

BOOST_AUTO_TEST_CASE(CreateSequenceSparseInGPU)
{
    if (ShouldRunOnGpu())
    {
        CreateSequenceTestSprse<float>(DeviceDescriptor::GPUDevice(0), false);
        CreateSequenceTestSprse<double>(DeviceDescriptor::GPUDevice(0), true);
    }
}

BOOST_AUTO_TEST_SUITE_END()

}}
