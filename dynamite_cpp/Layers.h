//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once
#include <exception>
#include <algorithm>
#include "CNTKLibrary.h"
#include <functional>
#include <fstream>
#include <random>

using namespace CNTK;

#pragma warning(push)
#pragma warning(disable: 4996)

//FunctionPtr FullyConnectedDNNLayer(Variable input, size_t outputDim, const DeviceDescriptor& device, const std::function<FunctionPtr(const FunctionPtr&)>& nonLinearity, const std::wstring& outputName = L"")
//{
//    return nonLinearity(FullyConnectedLinearLayer(input, outputDim, device, outputName));
//}

template <typename ElementType>
FunctionPtr Stabilize(const Variable& x, const DeviceDescriptor& device)
{
    ElementType scalarConstant = 4.0f;
    auto f = Constant::Scalar(scalarConstant);
    auto fInv = Constant::Scalar(f.GetDataType(), 1.0 / scalarConstant);

    auto beta = ElementTimes(fInv, Log(Constant::Scalar(f.GetDataType(), 1.0) + Exp(ElementTimes(f, Parameter({}, f.GetDataType(), 0.99537863 /* 1/f*ln (e^f-1) */, device)))));
    return ElementTimes(beta, x);
}

template <typename ElementType>
std::pair<FunctionPtr, FunctionPtr> LSTMPCellWithSelfStabilization(Variable input, Variable prevOutput, Variable prevCellState, const DeviceDescriptor& device)
{
    size_t outputDim = prevOutput.Shape()[0];
    size_t cellDim = prevCellState.Shape()[0];

    auto createBiasParam = [device](size_t dim) {
        return Parameter({ dim }, (ElementType)0.0, device);
    };

    unsigned long seed2 = 1;
    auto createProjectionParam = [device, &seed2](size_t outputDim) {
        return Parameter({ outputDim, NDShape::InferredDimension }, AsDataType<ElementType>(), GlorotUniformInitializer(1.0, 1, 0, seed2++), device);
    };

    auto createDiagWeightParam = [device, &seed2](size_t dim) {
        return Parameter({ dim }, AsDataType<ElementType>(), GlorotUniformInitializer(1.0, 1, 0, seed2++), device);
    };

    auto stabilizedPrevOutput = Stabilize<ElementType>(prevOutput, device);
    auto stabilizedPrevCellState = Stabilize<ElementType>(prevCellState, device);

    auto projectInput = [input, cellDim, createBiasParam, createProjectionParam]() {
        return createBiasParam(cellDim) + Times(createProjectionParam(cellDim), input);
    };

    // Input gate
    auto it = Sigmoid(projectInput() + Times(createProjectionParam(cellDim), stabilizedPrevOutput) + ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
    auto bit = ElementTimes(it, Tanh(projectInput() + Times(createProjectionParam(cellDim), stabilizedPrevOutput)));

    // Forget-me-not gate
    auto ft = Sigmoid(projectInput() + Times(createProjectionParam(cellDim), stabilizedPrevOutput) + ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
    auto bft = ElementTimes(ft, prevCellState);

    auto ct = bft + bit;

    // Output gate
    auto ot = Sigmoid(projectInput() + Times(createProjectionParam(cellDim), stabilizedPrevOutput) + ElementTimes(createDiagWeightParam(cellDim), Stabilize<ElementType>(ct, device)));
    auto ht = ElementTimes(ot, Tanh(ct));

    auto c = ct;
    auto h = (outputDim != cellDim) ? Times(createProjectionParam(outputDim), Stabilize<ElementType>(ht, device)) : ht;

    return{ h, c };
}

template <typename ElementType>
std::pair<FunctionPtr, FunctionPtr> LSTMPComponentWithSelfStabilization(Variable input,
    const NDShape& outputShape,
    const NDShape& cellShape,
    const std::function<FunctionPtr(const Variable&)>& recurrenceHookH,
    const std::function<FunctionPtr(const Variable&)>& recurrenceHookC,
    const DeviceDescriptor& device)
{
    auto dh = PlaceholderVariable(outputShape, input.DynamicAxes());
    auto dc = PlaceholderVariable(cellShape, input.DynamicAxes());

    auto LSTMCell = LSTMPCellWithSelfStabilization<ElementType>(input, dh, dc, device);

    auto actualDh = recurrenceHookH(LSTMCell.first);
    auto actualDc = recurrenceHookC(LSTMCell.second);

    // Form the recurrence loop by replacing the dh and dc placeholders with the actualDh and actualDc
    LSTMCell.first->ReplacePlaceholders({ { dh, actualDh }, { dc, actualDc } });

    return{ LSTMCell.first, LSTMCell.second };
}

// This is currently unused
inline FunctionPtr SimpleRecurrentLayer(const  Variable& input, const  NDShape& outputDim, const std::function<FunctionPtr(const Variable&)>& recurrenceHook, const DeviceDescriptor& device)
{
    auto dh = PlaceholderVariable(outputDim, input.DynamicAxes());

    unsigned long seed = 1;
    auto createProjectionParam = [device, &seed](size_t outputDim, size_t inputDim) {
        return Parameter(NDArrayView::RandomUniform<float>({ outputDim, inputDim }, -0.5, 0.5, seed++, device));
    };

    auto hProjWeights = createProjectionParam(outputDim[0], outputDim[0]);
    auto inputProjWeights = createProjectionParam(outputDim[0], input.Shape()[0]);

    auto output = Times(hProjWeights, recurrenceHook(dh)) + Times(inputProjWeights, input);
    return output->ReplacePlaceholders({ { dh, output } });
}
