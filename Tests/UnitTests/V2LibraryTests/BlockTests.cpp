//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"
#include <numeric>

static unsigned long seed = 1;

namespace CNTK { namespace Test {

FunctionPtr LinearLayerBlock(Variable input, size_t outputDim, const DeviceDescriptor& device, const std::wstring& outputName = L"")
{
    auto inputPlaceholder = PlaceholderVariable();

    auto timesParam = Parameter({ outputDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(DefaultParamInitScale, SentinelValueForInferParamInitRank, SentinelValueForInferParamInitRank, 1), device, L"timesParam");
    auto timesFunction = Times(timesParam, inputPlaceholder, L"times");

    auto plusParam = Parameter({ outputDim }, 0.0f, device, L"plusParam");
    return AsBlock(Plus(plusParam, timesFunction), { { inputPlaceholder, input } }, L"LinearLayer", outputName);
}

FunctionPtr SimpleRecurrentBlock(const Variable& prevOutput, const Variable& input, const DeviceDescriptor& device, const std::wstring& outputName = L"")
{
    assert(prevOutput.Shape().Rank() == 1);
    assert((prevOutput.Shape() != NDShape::Unknown) && !prevOutput.Shape().HasInferredDimension() && !prevOutput.Shape().HasFreeDimension());
    auto outputDim = prevOutput.Shape()[0];

    auto prevOutputPlaceholder = PlaceholderVariable();
    auto inputPlaceholder = PlaceholderVariable();
    auto prevOutputProjection = LinearLayerBlock(prevOutputPlaceholder, outputDim, device);
    auto inputProjection = LinearLayerBlock(inputPlaceholder, outputDim, device);
    return AsBlock(prevOutputProjection + inputProjection, { { prevOutputPlaceholder, prevOutput }, { inputPlaceholder, input } }, L"SimpleRecurrentBlock", outputName);
}

FunctionPtr SimpleRecurrentLayerBlock(const Variable& input, size_t outputDim, const DeviceDescriptor& device, const std::wstring& outputName = L"")
{
    auto placeholderOutput = PlaceholderVariable(NDShape(1, outputDim));
    auto inputPlaceholder = PlaceholderVariable();
    auto recurrenceBlock = SimpleRecurrentBlock(PastValue(placeholderOutput), inputPlaceholder, device);
    auto recurrentComposite = recurrenceBlock->ReplacePlaceholders({ { placeholderOutput, recurrenceBlock->Output()} });
    return AsBlock(std::move(recurrentComposite), { { inputPlaceholder, input } }, L"SimpleRecurrentLayer", outputName);
}

void TestBlocksWithRecurrence(size_t inputDim, size_t outputDim, const DeviceDescriptor& device)
{
    auto inputVar = InputVariable({ inputDim }, DataType::Float, true, L"input");

    auto inputPlaceholder = PlaceholderVariable();
    auto recurrentBlock1 = SimpleRecurrentLayerBlock(inputPlaceholder, outputDim, device);
    auto recurrentBlock2 = recurrentBlock1->Clone(ParameterCloningMethod::Share);
    auto recurrentBlock3 = recurrentBlock1->Clone(ParameterCloningMethod::Clone);

    auto recurrentBlock1Params = recurrentBlock1->Parameters();
    auto recurrentBlock2Params = recurrentBlock2->Parameters();
    auto recurrentBlock3Params = recurrentBlock3->Parameters();
    if (recurrentBlock1Params != recurrentBlock2Params)
        ReportFailure("The parameters of block created by cloning with parameter sharing are not same as the parameters of the clonee block");

    for (auto param : recurrentBlock3Params)
    {
        if (std::find(recurrentBlock1Params.begin(), recurrentBlock1Params.end(), param) != recurrentBlock1Params.end())
            ReportFailure("The parameters of block created by cloning with parameters cloned are still same as the parameters of the clonee block");
    }

    auto networkOutput = Sequence::Last(recurrentBlock1) + Sequence::Last(recurrentBlock2) + Sequence::Last(recurrentBlock3);
    networkOutput->ReplacePlaceholders({ { recurrentBlock1->Arguments()[0], inputVar }, { recurrentBlock2->Arguments()[0], inputVar }, { recurrentBlock3->Arguments()[0], inputVar } });

    srand(seed);

    size_t numSequences = 5;
    size_t maxAllowedSequenceLength = 17;
    std::vector<size_t> sequenceLengths(numSequences);
    size_t maxActualSequenceLength = 0;
    for (size_t i = 0; i < numSequences; ++i)
    {
        sequenceLengths[i] = (rand() % maxAllowedSequenceLength) + 1;
        if (sequenceLengths[i] > maxActualSequenceLength)
            maxActualSequenceLength = sequenceLengths[i];
    }

    NDShape inputShape = inputVar.Shape().AppendShape({ maxActualSequenceLength, numSequences });
    ValuePtr inputValue;
    size_t totalNumInputSamples = maxActualSequenceLength * numSequences;
    std::vector<float> inputData(inputDim * totalNumInputSamples, std::numeric_limits<float>::quiet_NaN());
    for (size_t i = 0; i < numSequences; ++i)
    {
        for (size_t j = 0; j < maxActualSequenceLength; ++j)
        {
            size_t sampleIdx = (i * maxActualSequenceLength) + j;
            for (size_t k = 0; k < inputDim; ++k)
            {
                if (j < sequenceLengths[i])
                    inputData[(sampleIdx * inputDim) + k] = ((float)rand()) / RAND_MAX;
            }
        }
    }

    NDArrayViewPtr inputValueData = MakeSharedObject<NDArrayView>(inputShape, inputData.data(), inputData.size(), DeviceDescriptor::CPUDevice(), true);
    NDMaskPtr inputMask = MakeSharedObject<NDMask>(NDShape({ maxActualSequenceLength, numSequences }));
    for (size_t i = 0; i < numSequences; ++i)
    {
        inputMask->MarkSequenceBegin({ 0, i });
        inputMask->InvalidateSection({ sequenceLengths[i], i }, { NDShape::InferredDimension, 1 });
    }

    inputValue = MakeSharedObject<Value>(inputValueData, inputMask);

    std::unordered_map<Variable, ValuePtr> outputs = { { networkOutput, nullptr } };
    networkOutput->Forward({ { inputVar, inputValue } }, outputs, device);
    auto outputValue = outputs[networkOutput];
    if (outputValue->Shape()[0] != outputDim)
        ReportFailure("Output value shape's leading dimensions does not match expected output dim (%d)", (int)outputDim);
}

BOOST_AUTO_TEST_SUITE(BlockSuite)

BOOST_AUTO_TEST_CASE(BlocksWithRecurrence)
{
    if (ShouldRunOnCpu())
        TestBlocksWithRecurrence(7, 5, DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_CASE(ChangingParameterValuesInGPU)
{
    if (ShouldRunOnGpu())
        TestBlocksWithRecurrence(11, 15, DeviceDescriptor::GPUDevice(0));
}

BOOST_AUTO_TEST_SUITE_END()

}}
