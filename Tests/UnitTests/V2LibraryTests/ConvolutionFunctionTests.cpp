//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include <boost/test/unit_test.hpp>

#include <exception>
#include <algorithm>
#include "CNTKLibrary.h"
#include <functional>
#include <fstream>
#include <random>
#include "stdafx.h"
#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"
#include <numeric>
#include "CNTKLibraryC.h"

using namespace CNTK;

namespace CNTK
{
namespace Test
{

static unsigned long seed = 1;

/// <summary>
/// Print out the evalaution results.
/// </summary>
template <typename ElementType>
void PrintOutput(size_t sampleSize, std::vector<std::vector<ElementType>> outputBuffer)
{
    printf("The batch contains %d sequences.\n", (int) outputBuffer.size());
    for (size_t seqNo = 0; seqNo < outputBuffer.size(); seqNo++)
    {
        auto seq = outputBuffer[seqNo];
        if (seq.size() % sampleSize != 0)
            throw("The number of elements in the sequence is not a multiple of sample size");

        printf("Sequence %d contains %d samples.\n", (int) seqNo, (int) (seq.size() / sampleSize));
        size_t sampleNo = 0;
        for (size_t i = 0; i < seq.size();)
        {
            if (i % sampleSize == 0)
                printf("    sample %d: ", (int) sampleNo);
            printf("%f", seq[i++]);
            if (i % sampleSize == 0)
            {
                printf(".\n");
                sampleNo++;
            }
            else
                printf(", ");
        }
    }
}

template <typename ElementType>
ValuePtr CreateBatchWithVariableSequence(const std::vector<size_t>& sampleSizes, size_t batchSize, const std::vector<size_t>& sequenceSize, const std::vector<ElementType>& batchData, const DeviceDescriptor& device, bool readOnly = false, bool sequential = false)
{
    if (sequenceSize.size() != batchSize)
        InvalidArgument("The number of sequences (%zu) in the vector containing sequence size must match batch size (%zu)", sequenceSize.size(), batchSize);

    std::vector<NDArrayViewPtr> sequencesView(batchSize);
    size_t curBatchDataIdx = 0;
    for (size_t i = 0; i < batchSize; i++)
    {
        auto sampleShape = NDShape({sampleSizes[i]});
        if (sequential)
            sampleShape = sampleShape.AppendShape({1});
        auto sequenceDataShape = sampleShape.AppendShape({sequenceSize[i]});
        sequencesView[i] = MakeSharedObject<NDArrayView>(sequenceDataShape, batchData.data() + curBatchDataIdx, sampleSizes[i] * sequenceSize[i], DeviceDescriptor::CPUDevice());
        curBatchDataIdx += sampleSizes[i] * sequenceSize[i];
    }

    auto sampleShape = NDShape({sampleSizes[0]});
    if (sequential)
        sampleShape = sampleShape.AppendShape({1});

    return Value::Create(sampleShape, sequencesView, {}, device, readOnly, true);
}

template <typename ElementType>
ValuePtr CreateBatchWithVariableSequence(const NDShape& sampleShape, size_t batchSize, const std::vector<size_t>& sequenceSize, const std::vector<ElementType>& batchData, const DeviceDescriptor& device, bool readOnly = false)
{
    auto shapeSize = sampleShape.TotalSize();
    if (batchData.size() % shapeSize != 0)
        InvalidArgument("The number of elements (%zu) in the vector containing batch data must be a multiple of the size (%zu) of the sample shape '%S'.",
                        batchData.size(), shapeSize, sampleShape.AsString().c_str());

    if (sequenceSize.size() != batchSize)
        InvalidArgument("The number of sequences (%zu) in the vector containing sequence size must match batch size (%zu)", sequenceSize.size(), batchSize);

    std::vector<NDArrayViewPtr> sequencesView(batchSize);
    size_t curBatchDataIdx = 0;
    for (size_t i = 0; i < batchSize; i++)
    {
        auto sequenceDataShape = sampleShape.AppendShape({sequenceSize[i]});
        sequencesView[i] = MakeSharedObject<NDArrayView>(sequenceDataShape, batchData.data() + curBatchDataIdx, shapeSize * sequenceSize[i], DeviceDescriptor::CPUDevice());
        curBatchDataIdx += shapeSize * sequenceSize[i];
    }

    return Value::Create(sampleShape, sequencesView, {}, device, readOnly, true);
}

template <typename ElementType>
void Run1DFreeDimConvLayer(const DeviceDescriptor& device, bool testFreeDimension = true)
{
    auto input = InputVariable({NDShape::FreeDimension}, AsDataType<ElementType>(), L"features");
    if (!testFreeDimension)
    {
        input = InputVariable({10}, AsDataType<ElementType>(), L"features");
    }
    auto convParam = Parameter({3, 1}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    auto conv = Convolution(convParam, input, {2}, {true}, {true}, {1}, 0);
    auto convb = Parameter({1}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    auto relu = LeakyReLU(Plus(conv, convb), 0.01);

    const size_t inputDataSize = 10;
    const size_t channelSize = 1;
    const std::vector<size_t> sequenceSize = {3, 2, 1, 1};
    const size_t batchSize = sequenceSize.size();
    const std::vector<size_t> sampleSizes = {10, 10, 10, 10};
    size_t dataSize = 0;
    for (size_t i = 0; i < sequenceSize.size(); i++)
    {
        dataSize += sampleSizes[i] * sequenceSize[i];
    }

    std::vector<ElementType> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i)
    {
        inputData[i] = static_cast<ElementType>(i % 255);
    }

    auto inputVal = CreateBatchWithVariableSequence(sampleSizes, batchSize, sequenceSize, inputData, device);
    auto outputVar = relu->Output();
    std::unordered_map<Variable, ValuePtr> inputDataMap = {{input, inputVal}};

    std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

    relu->Evaluate(inputDataMap, outputDataMap, device);

    auto outputVal = outputDataMap[outputVar];
    std::vector<std::vector<ElementType>> outputData;

    Internal::SetAutomaticUnpackingOfPackedValues(false);
    outputVal->CopyVariableValueTo(outputVar, outputData);
    Internal::SetAutomaticUnpackingOfPackedValues(true);

    PrintOutput<ElementType>(inputDataSize / 2, outputData);
}

template <typename ElementType>
void Run1DFreeDimSimpConvLayer(const DeviceDescriptor& device, bool testFreeDimension = true)
{
    auto input = InputVariable({NDShape::FreeDimension}, AsDataType<ElementType>(), L"features");
    if (!testFreeDimension)
    {
        input = InputVariable({10}, AsDataType<ElementType>(), L"features");
    }
    auto convParam = Parameter({3, 1}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    auto conv = Convolution(convParam, input, {2}, { true }, { true }, { 1 }, 0);
    auto convParam2 = Parameter({2, 1}, AsDataType<ElementType>(), (ElementType) 0.5f, device);
    auto conv2 = Convolution(convParam2, conv, {2});

    const size_t inputDataSize = 10;
    const size_t channelSize = 1;
    const std::vector<size_t> sequenceSize = {2, 3, 1, 1};
    const size_t batchSize = sequenceSize.size();
    const std::vector<size_t> sampleSizes = {10, 10, 10, 10};
    size_t dataSize = 0;
    for (size_t i = 0; i < sequenceSize.size(); i++)
    {
        dataSize += sampleSizes[i] * sequenceSize[i];
    }

    std::vector<ElementType> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i)
    {
        inputData[i] = static_cast<ElementType>(i % 255);
    }

    auto inputVal = CreateBatchWithVariableSequence(sampleSizes, batchSize, sequenceSize, inputData, device);
    auto outputVar = conv2->Output();
    std::unordered_map<Variable, ValuePtr> inputDataMap = {{input, inputVal}};

    std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

    conv2->Evaluate(inputDataMap, outputDataMap, device);

    auto outputVal = outputDataMap[outputVar];
    std::vector<std::vector<ElementType>> outputData;

    Internal::SetAutomaticUnpackingOfPackedValues(false);
    outputVal->CopyVariableValueTo(outputVar, outputData);
    Internal::SetAutomaticUnpackingOfPackedValues(true);

    PrintOutput<ElementType>(inputDataSize / 2 / 2 + 1, outputData);
}

template <typename ElementType>
void Run1DSeqConvLayer(const DeviceDescriptor& device, bool auto_padding = true)
{
    auto input = InputVariable({1}, AsDataType<ElementType>(), L"features");
    auto convParam = Parameter({3, 1}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    //auto conv = Convolution(convParam, input, {2, 1}, {true}, {true}, true);
    // test auto fixing filter shape
    auto conv = Convolution(convParam, input, {
                                                  2,
                                              },
                            {true}, {auto_padding}, { 1 }, 1, 1, 0, true);
    auto convb = Parameter({1}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    auto relu = LeakyReLU(Plus(conv, convb), 0.01);

    const std::vector<size_t> sequenceSize = {5, 10, 8, 4};
    const size_t batchSize = sequenceSize.size();
    size_t dataSize = 0;
    for (size_t i = 0; i < sequenceSize.size(); i++)
    {
        dataSize += sequenceSize[i];
    }

    std::vector<ElementType> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i)
    {
        inputData[i] = static_cast<ElementType>(i % 255);
    }

    auto sampleShape = NDShape({1});

    auto inputVal = CreateBatchWithVariableSequence(sampleShape, batchSize, sequenceSize, inputData, device);
    auto outputVar = relu->Output();
    std::unordered_map<Variable, ValuePtr> inputDataMap = {{input, inputVal}};

    std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

    relu->Evaluate(inputDataMap, outputDataMap, device);

    auto outputVal = outputDataMap[outputVar];
    std::vector<std::vector<ElementType>> outputData;

    Internal::SetAutomaticUnpackingOfPackedValues(false);
    outputVal->CopyVariableValueTo(outputVar, outputData);
    Internal::SetAutomaticUnpackingOfPackedValues(true);

    PrintOutput<ElementType>(1, outputData);
}

template <typename ElementType>
std::vector<std::vector<ElementType>> RunConvSeqByUnpack_byhand(const DeviceDescriptor& device)
{
    const size_t numFilters = 2;

    auto input = InputVariable({20, 1}, AsDataType<ElementType>(), L"features");
    auto input_ = Reshape(input, {10, 2});
    auto convParam = Parameter({3, 2, 2}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    auto unpackInputOutputs = Sequence::Unpack(input_, (ElementType) 0.0f, false, L"unpack input");
    auto unpackInput = unpackInputOutputs->Outputs()[0];
    auto unpackInputMask = unpackInputOutputs->Outputs()[1];
    auto transposeInput = TransposeAxes(unpackInput, Axis(-1), Axis(-2), L"transpose axis input");
    auto conv = Convolution(convParam, transposeInput, {2, 2}); //auto conv = Convolution(convParam, input, {2, 2}, /*sharing = */ {true}, /*autoPadding = */ {true}, /*sequential = */ true);

    auto unpackInputMaskReduceSum = ReduceSum(unpackInputMask, Axis(-1));
    auto seqKernelSize = convParam.Shape()[convParam.Shape().Rank() - 2];
    auto convOutputSeqSize = Ceil(ElementDivide(unpackInputMaskReduceSum, Constant::Scalar((ElementType) seqKernelSize)));

    auto transOut = TransposeAxes(conv, Axis(-1), Axis(-2), L"transpose axis output");

    auto resPack = ToSequence(transOut, convOutputSeqSize, L"pack output axis", L"pack output"); // provide sequence length as parameter.

    const size_t channelSize = 2;
    const std::vector<size_t> sequenceSize = {4, 5, 2, 2};
    const size_t batchSize = sequenceSize.size();
    const std::vector<size_t> sampleSizes = {10 * channelSize, 10 * channelSize, 10 * channelSize, 10 * channelSize };
    size_t dataSize = 0;
    for (size_t i = 0; i < sequenceSize.size(); i++)
    {
        dataSize += sampleSizes[i] * sequenceSize[i];
    }

    std::vector<ElementType> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i)
    {
        inputData[i] = static_cast<ElementType>(i % 255);
    }

    auto inputVal = CreateBatchWithVariableSequence(sampleSizes, batchSize, sequenceSize, inputData, device, false, true);
    auto outputVar = resPack->Output();
    std::unordered_map<Variable, ValuePtr> inputDataMap = {{input, inputVal}};

    std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

    resPack->Evaluate(inputDataMap, outputDataMap, device);

    auto outputVal = outputDataMap[outputVar];
    std::vector<std::vector<ElementType>> outputData;

    Internal::SetAutomaticUnpackingOfPackedValues(false);
    outputVal->CopyVariableValueTo(outputVar, outputData);
    Internal::SetAutomaticUnpackingOfPackedValues(true);

    PrintOutput<ElementType>(5, outputData);

    return outputData;
}

template <typename ElementType>
std::vector<std::vector<ElementType>> RunConvSeqByUnpack(const DeviceDescriptor& device)
{
    const size_t numFilters = 2;

    auto input = InputVariable({20, 1}, AsDataType<ElementType>(), L"features");
    auto input_ = Reshape(input, {10, 2});
    auto convParam = Parameter({3, 2, 2}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    auto conv = Convolution(convParam, input_, {2, 2}, /*sharing = */ {true}, /*autoPadding = */ {true}, { 1 }, 1, 1, 0, true);

    const size_t channelSize = 2;
    const std::vector<size_t> sequenceSize = {4, 5, 2, 2};
    const size_t batchSize = sequenceSize.size();
    const std::vector<size_t> sampleSizes = {10 * channelSize, 10 * channelSize, 10 * channelSize, 10 * channelSize };
    size_t dataSize = 0;
    for (size_t i = 0; i < sequenceSize.size(); i++)
    {
        dataSize += sampleSizes[i] * sequenceSize[i];
    }

    std::vector<ElementType> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i)
    {
        inputData[i] = static_cast<ElementType>(i % 255);
    }

    auto inputVal = CreateBatchWithVariableSequence(sampleSizes, batchSize, sequenceSize, inputData, device, false, true);
    auto outputVar = conv->Output();
    std::unordered_map<Variable, ValuePtr> inputDataMap = {{input, inputVal}};

    std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

    conv->Evaluate(inputDataMap, outputDataMap, device);

    auto outputVal = outputDataMap[outputVar];
    std::vector<std::vector<ElementType>> outputData;

    Internal::SetAutomaticUnpackingOfPackedValues(false);
    outputVal->CopyVariableValueTo(outputVar, outputData);
    Internal::SetAutomaticUnpackingOfPackedValues(true);

    PrintOutput<ElementType>(5, outputData);

    return outputData;
}

template <typename ElementType>
std::vector<std::vector<ElementType>> RunConvSeqByUnpackTestMaskReduce(const DeviceDescriptor& device)
{
    auto input = InputVariable({20, 1}, AsDataType<ElementType>(), L"features");
    auto input_ = Reshape(input, {10, 2});
    auto convParam = Parameter({3, 2, 2}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    auto unpackInputOutputs = Sequence::Unpack(input_, (ElementType) 0.0f, false, L"unpack input");
    auto unpackInputMask = unpackInputOutputs->Outputs()[1]; // TODO : can we compute output mask by input mask, and convert output back to sequence using mask?

    auto unpackInputMaskReduceSum = ReduceSum(unpackInputMask, Axis(-1));
    auto seqKernelSize = convParam.Shape()[convParam.Shape().Rank() - 2];

    auto convOutputSeqSize = Ceil(ElementDivide(unpackInputMaskReduceSum, Constant::Scalar((ElementType) seqKernelSize)));

    const size_t channelSize = 2;
    const std::vector<size_t> sequenceSize = {4, 5, 2, 2};
    const size_t batchSize = sequenceSize.size();
    const std::vector<size_t> sampleSizes = {10 * channelSize, 10 * channelSize, 10 * channelSize, 10 * channelSize };
    size_t dataSize = 0;
    for (size_t i = 0; i < sequenceSize.size(); i++)
    {
        dataSize += sampleSizes[i] * sequenceSize[i];
    }

    std::vector<ElementType> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i)
    {
        inputData[i] = static_cast<ElementType>(i % 255);
    }

    auto inputVal = CreateBatchWithVariableSequence(sampleSizes, batchSize, sequenceSize, inputData, device, false, true);
    auto outputVar = convOutputSeqSize->Output();
    std::unordered_map<Variable, ValuePtr> inputDataMap = {{input, inputVal}};

    std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

    convOutputSeqSize->Evaluate(inputDataMap, outputDataMap, device);

    auto outputVal = outputDataMap[outputVar];
    std::vector<std::vector<ElementType>> outputData;

    Internal::SetAutomaticUnpackingOfPackedValues(false);
    outputVal->CopyVariableValueTo(outputVar, outputData);
    Internal::SetAutomaticUnpackingOfPackedValues(true);

    PrintOutput<ElementType>(1, outputData);

    return outputData;
}

template <typename ElementType>
std::vector<std::vector<ElementType>> RunConvMatchResSeqByUnpack(const DeviceDescriptor& device)
{
    auto input = InputVariable({10, 2, 5}, AsDataType<ElementType>(), L"features");
    auto input_ = TransposeAxes(input, Axis(1), Axis(2));
    auto convParam = Parameter({3, 2, 2}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    auto conv = Convolution(convParam, input_, {2, 2});

    const size_t channelSize = 2;
    const std::vector<size_t> sequenceSize = {4, 5, 2, 2};
    const size_t batchSize = sequenceSize.size();
    const std::vector<size_t> sampleSizes = {10 * channelSize, 10 * channelSize, 10 * channelSize, 10 * channelSize};
    size_t dataSize = 0;
    for (size_t i = 0; i < sequenceSize.size(); i++)
    {
        dataSize += sampleSizes[i] * 5;
    }

    std::vector<ElementType> inputData(dataSize);
    size_t k = 0;
    size_t l = 0;
    for (size_t i = 0; i < sequenceSize.size(); ++i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t z = 0; z < 20; ++z)
            {
                if (j >= sequenceSize[i])
                    inputData[k] = static_cast<ElementType>(0);
                else
                {
                    inputData[k] = static_cast<ElementType>(l % 255);
                    l++;
                }
                k++;
            }
        }
    }

    //const std::vector<size_t> sampleSizes_ = {50, 50, 50, 50};

    auto inputVal = CreateBatchWithVariableSequence(NDShape({10, 2, 5}), batchSize, {1, 1, 1, 1}, inputData, device);
    auto outputVar = conv->Output();
    std::unordered_map<Variable, ValuePtr> inputDataMap = {{input, inputVal}};

    std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

    conv->Evaluate(inputDataMap, outputDataMap, device);

    auto outputVal = outputDataMap[outputVar];
    std::vector<std::vector<ElementType>> outputData;

    Internal::SetAutomaticUnpackingOfPackedValues(false);
    outputVal->CopyVariableValueTo(outputVar, outputData);
    Internal::SetAutomaticUnpackingOfPackedValues(true);

    PrintOutput<ElementType>(15, outputData);

    return outputData;
}

template <typename ElementType>
void RunConvRankTests1(const DeviceDescriptor& device)
{
    auto input_ = InputVariable({12}, AsDataType<ElementType>());
    auto input = Reshape(input_, {4, 3, 1});
    auto params = Parameter({3, 2, 1}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    // requires kernel dim <= input dim.

    auto conv = Convolution(params, input, {2, 2});

    const size_t inputDataSize = 12;
    const std::vector<size_t> sequenceSize = {2, 3};
    const size_t batchSize = sequenceSize.size();
    const std::vector<size_t> sampleSizes = {inputDataSize, inputDataSize};
    size_t dataSize = 0;
    for (size_t i = 0; i < sequenceSize.size(); i++)
    {
        dataSize += sampleSizes[i] * sequenceSize[i];
    }

    std::vector<ElementType> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i)
    {
        inputData[i] = static_cast<ElementType>(i % 255);
    }

    auto inputVal = CreateBatchWithVariableSequence(sampleSizes, batchSize, sequenceSize, inputData, device);
    auto outputVar = conv->Output();
    std::unordered_map<Variable, ValuePtr> inputDataMap = {{input_, inputVal}};

    std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

    conv->Evaluate(inputDataMap, outputDataMap, device);

    auto outputVal = outputDataMap[outputVar];
    std::vector<std::vector<ElementType>> outputData;

    Internal::SetAutomaticUnpackingOfPackedValues(false);
    outputVal->CopyVariableValueTo(outputVar, outputData);
    Internal::SetAutomaticUnpackingOfPackedValues(true);

    PrintOutput<ElementType>(4, outputData);
}

template <typename ElementType>
void RunConvRankTests2(const DeviceDescriptor& device)
{
    auto input_ = InputVariable({36}, AsDataType<ElementType>());
    auto input = Reshape(input_, {4, 3, 3});
    auto params = Parameter({3, 3, 2, 4}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    // requires kernel dim >= input dim ....

    auto conv = Convolution(params, input, {2, 2}, {true}, {true, true, true}, {1}, 0);

    const size_t inputDataSize = 36;
    const std::vector<size_t> sequenceSize = {2, 3};
    const size_t batchSize = sequenceSize.size();
    const std::vector<size_t> sampleSizes = {inputDataSize, inputDataSize};
    size_t dataSize = 0;
    for (size_t i = 0; i < sequenceSize.size(); i++)
    {
        dataSize += sampleSizes[i] * sequenceSize[i];
    }

    std::vector<ElementType> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i)
    {
        inputData[i] = static_cast<ElementType>(i % 255);
    }

    auto inputVal = CreateBatchWithVariableSequence(sampleSizes, batchSize, sequenceSize, inputData, device);
    auto outputVar = conv->Output();
    std::unordered_map<Variable, ValuePtr> inputDataMap = {{input_, inputVal}};

    std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

    conv->Evaluate(inputDataMap, outputDataMap, device);

    auto outputVal = outputDataMap[outputVar];
    std::vector<std::vector<ElementType>> outputData;

    Internal::SetAutomaticUnpackingOfPackedValues(false);
    outputVal->CopyVariableValueTo(outputVar, outputData);
    Internal::SetAutomaticUnpackingOfPackedValues(true);

    PrintOutput<ElementType>(32, outputData);
}

template <typename ElementType>
void RunConvRankTests3(const DeviceDescriptor& device)
{
    auto input_ = InputVariable({6}, AsDataType<ElementType>());
    auto input = Reshape(input_, {6});
    auto params = Parameter({2, 4}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    // requires kernel dim >= input dim ....

    auto conv = Convolution(params, input, {5}, { true }, { true }, { 1 }, 0);

    const size_t inputDataSize = 6;
    const std::vector<size_t> sequenceSize = {2, 3};
    const size_t batchSize = sequenceSize.size();
    const std::vector<size_t> sampleSizes = {inputDataSize, inputDataSize};
    size_t dataSize = 0;
    for (size_t i = 0; i < sequenceSize.size(); i++)
    {
        dataSize += sampleSizes[i] * sequenceSize[i];
    }

    std::vector<ElementType> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i)
    {
        inputData[i] = static_cast<ElementType>(i % 255);
    }

    auto inputVal = CreateBatchWithVariableSequence(sampleSizes, batchSize, sequenceSize, inputData, device);
    auto outputVar = conv->Output();
    std::unordered_map<Variable, ValuePtr> inputDataMap = {{input_, inputVal}};

    std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

    conv->Evaluate(inputDataMap, outputDataMap, device);

    auto outputVal = outputDataMap[outputVar];
    std::vector<std::vector<ElementType>> outputData;

    Internal::SetAutomaticUnpackingOfPackedValues(false);
    outputVal->CopyVariableValueTo(outputVar, outputData);
    Internal::SetAutomaticUnpackingOfPackedValues(true);

    PrintOutput<ElementType>(8, outputData);
}

template <typename ElementType>
void RunConvRankTests4(const DeviceDescriptor& device)
{
    auto input_ = InputVariable({9}, AsDataType<ElementType>());
    auto input = Reshape(input_, {3, 3});
    auto params = Parameter({2, 3, 4}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    // requires kernel dim >= input dim ....

    auto conv = Convolution(params, input, {2}, { true }, { true, true }, { 1 }, 0);

    const size_t inputDataSize = 9;
    const std::vector<size_t> sequenceSize = {2, 3};
    const size_t batchSize = sequenceSize.size();
    const std::vector<size_t> sampleSizes = {inputDataSize, inputDataSize};
    size_t dataSize = 0;
    for (size_t i = 0; i < sequenceSize.size(); i++)
    {
        dataSize += sampleSizes[i] * sequenceSize[i];
    }

    std::vector<ElementType> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i)
    {
        inputData[i] = static_cast<ElementType>(i % 255);
    }

    auto inputVal = CreateBatchWithVariableSequence(sampleSizes, batchSize, sequenceSize, inputData, device);
    auto outputVar = conv->Output();
    std::unordered_map<Variable, ValuePtr> inputDataMap = {{input_, inputVal}};

    std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

    conv->Evaluate(inputDataMap, outputDataMap, device);

    auto outputVal = outputDataMap[outputVar];
    std::vector<std::vector<ElementType>> outputData;

    Internal::SetAutomaticUnpackingOfPackedValues(false);
    outputVal->CopyVariableValueTo(outputVar, outputData);
    Internal::SetAutomaticUnpackingOfPackedValues(true);

    PrintOutput<ElementType>(16, outputData);
}

template <typename ElementType>
void RunConvDilTest(const DeviceDescriptor& device, bool sequential = false)
{
    auto input_ = InputVariable({12}, AsDataType<ElementType>());
    // The extra (1) here is crucial as we don't support omitting channel size 1 currently.
    auto input = Reshape(input_, {3, 4, 1});

    if (sequential)
    {
        input_ = InputVariable({3}, AsDataType<ElementType>());
        input = Reshape(input_, {3, 1});
    }

    auto params = Parameter({2, 3, 1, 2}, AsDataType<ElementType>(), (ElementType) 1.0f, device);

    auto conv = Convolution(params, input, {1}, {true}, {true}, {1, 2}, 1, 1, 0, sequential);

    size_t inputDataSize = 12;
    std::vector<size_t> sequenceSize(5, 1);
    std::vector<size_t> sampleSizes(5 * 1, inputDataSize);

    if (sequential)
    {
        inputDataSize = 3;
        sequenceSize = std::vector<size_t>(5, 4);
        sampleSizes = std::vector<size_t>(5 * 4, inputDataSize);
    }

    const size_t batchSize = sequenceSize.size();

    size_t dataSize = 0;
    for (size_t i = 0; i < sequenceSize.size(); i++)
    {
        dataSize += sampleSizes[i] * sequenceSize[i];
    }
    std::vector<ElementType> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i)
    {
        inputData[i] = static_cast<ElementType>(i % 255);
    }

    auto inputVal = CreateBatchWithVariableSequence(sampleSizes, batchSize, sequenceSize, inputData, device);
    auto outputVar = conv->Output();
    std::unordered_map<Variable, ValuePtr> inputDataMap = {{input_, inputVal}};

    std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

    conv->Evaluate(inputDataMap, outputDataMap, device);

    auto outputVal = outputDataMap[outputVar];
    std::vector<std::vector<ElementType>> outputData;

    Internal::SetAutomaticUnpackingOfPackedValues(false);
    outputVal->CopyVariableValueTo(outputVar, outputData);
    Internal::SetAutomaticUnpackingOfPackedValues(true);

    if (!sequential)
        PrintOutput<ElementType>(24, outputData);
    else
        PrintOutput<ElementType>(6, outputData);
}

BOOST_AUTO_TEST_SUITE(ConvolutionFunctionSuite)

BOOST_AUTO_TEST_CASE(ConvolutionNetworkDifferentRankInCPU)
{
    if (ShouldRunOnCpu())
    {
        auto device = DeviceDescriptor::CPUDevice();
        RunConvRankTests1<float>(device);
        RunConvRankTests2<float>(device);
        RunConvRankTests3<float>(device);
        RunConvRankTests4<float>(device);
    }
}

BOOST_AUTO_TEST_CASE(ConvolutionNetwork1DFreeDimensionInCPU)
{
    if (ShouldRunOnCpu())
    {
        auto device = DeviceDescriptor::CPUDevice();
        Run1DFreeDimConvLayer<float>(device);
        Run1DFreeDimSimpConvLayer<float>(device);
    }
}

BOOST_AUTO_TEST_CASE(ConvolutionNetwork1DSequentialInCPU)
{
    if (ShouldRunOnCpu())
    {
        auto device = DeviceDescriptor::CPUDevice();
        Run1DSeqConvLayer<float>(device);
        Run1DSeqConvLayer<float>(device, false);
    }
}

BOOST_AUTO_TEST_CASE(ConvolutionNetworkSequentialValueTestInCPU)
{
    if (ShouldRunOnCpu())
    {
        auto device = DeviceDescriptor::CPUDevice();
        auto output_byhand = RunConvSeqByUnpack_byhand<float>(device);
        auto output_seq = RunConvSeqByUnpack<float>(device);
        auto output_seqSize = RunConvSeqByUnpackTestMaskReduce<float>(device);
        auto output_nonseq = RunConvMatchResSeqByUnpack<float>(device);

        BOOST_TEST(output_byhand.size() == output_seq.size());
        for (auto i = 0; i < output_byhand.size(); ++i)
        {
            BOOST_CHECK_EQUAL_COLLECTIONS(output_byhand[i].begin(), output_byhand[i].end(),
                                          output_seq[i].begin(), output_seq[i].end());
        }

        BOOST_TEST(output_seqSize.size() == 4);
        auto resSize = std::vector<float>{2.0, 3.0, 1.0, 1.0};
        for (auto i = 0; i < 4; ++i)
        {
            BOOST_TEST(output_seqSize[i][0] == resSize[i]);
        }

        BOOST_TEST(output_seq.size() == output_nonseq.size());
        for (auto i = 0; i < output_seq.size(); ++i)
        {
            BOOST_CHECK_EQUAL_COLLECTIONS(output_seq[i].begin(), output_seq[i].end(),
                                          output_nonseq[i].begin(), output_nonseq[i].begin() + output_seq[i].size());
        }
    }
}

BOOST_AUTO_TEST_CASE(ConvolutionNetworkDifferentRankInGPU)
{
    if (ShouldRunOnGpu())
    {
        auto device = DeviceDescriptor::GPUDevice(0);
        RunConvRankTests1<float>(device);
        RunConvRankTests2<float>(device);
        RunConvRankTests3<float>(device);
        RunConvRankTests4<float>(device);
    }
}

BOOST_AUTO_TEST_CASE(ConvolutionNetwork1DFreeDimensionInGPU)
{
    if (ShouldRunOnGpu())
    {
        auto device = DeviceDescriptor::GPUDevice(0);
        Run1DFreeDimConvLayer<float>(device);
        Run1DFreeDimSimpConvLayer<float>(device);
    }
}

BOOST_AUTO_TEST_CASE(ConvolutionNetwork1DSequentialInGPU)
{
    if (ShouldRunOnGpu())
    {
        auto device = DeviceDescriptor::GPUDevice(0);
        Run1DSeqConvLayer<float>(device);
        Run1DSeqConvLayer<float>(device, false);
    }
}

BOOST_AUTO_TEST_CASE(ConvolutionNetworkDilationInGPU)
{
    if (ShouldRunOnGpu())
    {
        auto device = DeviceDescriptor::GPUDevice(0);
        RunConvDilTest<float>(device);
    }
}

BOOST_AUTO_TEST_CASE(ConvolutionNetworkDilationSequentialInGPU)
{
    if (ShouldRunOnGpu())
    {
        auto device = DeviceDescriptor::GPUDevice(0);
        RunConvDilTest<float>(device, true);
    }
}

BOOST_AUTO_TEST_CASE(ConvolutionNetworkSequentialValueTestInGPU)
{
    if (ShouldRunOnGpu())
    {
        auto device = DeviceDescriptor::GPUDevice(0);
        auto output_byhand = RunConvSeqByUnpack_byhand<float>(device);
        auto output_seq = RunConvSeqByUnpack<float>(device);
        auto output_seqSize = RunConvSeqByUnpackTestMaskReduce<float>(device);
        auto output_nonseq = RunConvMatchResSeqByUnpack<float>(device);

        BOOST_TEST(output_byhand.size() == output_seq.size());
        for (auto i = 0; i < output_byhand.size(); ++i)
        {
            BOOST_CHECK_EQUAL_COLLECTIONS(output_byhand[i].begin(), output_byhand[i].end(),
                                          output_seq[i].begin(), output_seq[i].end());
        }

        BOOST_TEST(output_seqSize.size() == 4);
        auto resSize = std::vector<float>{2.0, 3.0, 1.0, 1.0};
        for (auto i = 0; i < 4; ++i)
        {
            BOOST_TEST(output_seqSize[i][0] == resSize[i]);
        }

        BOOST_TEST(output_seq.size() == output_nonseq.size());
        for (auto i = 0; i < output_seq.size(); ++i)
        {
            BOOST_CHECK_EQUAL_COLLECTIONS(output_seq[i].begin(), output_seq[i].end(),
                                          output_nonseq[i].begin(), output_nonseq[i].begin() + output_seq[i].size());
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
}
}