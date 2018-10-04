//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "proto/onnx/core/graph/model.h"

#include "RNNHelper.h"
#include "Operators.h"
#include "Utils.h"


using namespace CNTK;
using namespace CNTK::ONNX;
using namespace Microsoft::MSR::CNTK;

std::string MapActivationNameONNXToCNTK(const std::string &onnxOp)
{
    if (onnxOp == "Relu")
        return "ReLU";
    else if (onnxOp == "Sigmoid")
        return "StableSigmoid";
    else if (onnxOp == "LeakyRelu")
        return "LeakyReLU";
    else if (onnxOp == "ThresholdedRelu")
        return "ThresholdedReLU";
    else if (onnxOp == "Elu")
        return "ELU";
    else
        return onnxOp;
}

std::string MapActivationNameCNTKToONNX(const std::string &cntkOp)
{
    if (cntkOp == "ReLU")
        return "Relu";
    else if (cntkOp == "StableSigmoid")
        return "Sigmoid";
    else if (cntkOp == "LeakyReLU")
        return "LeakyRelu";
    else if (cntkOp == "ThresholdedReLU")
        return "ThresholdedRelu";
    else if (cntkOp == "ELU")
        return "Elu";
    else
        return cntkOp;
}

bool IsActivationOp(const std::string &activationName)
{
    return activationName == "Relu" || activationName == "ReLU" ||
        activationName == "Tanh" ||
        activationName == "Sigmoid" || activationName == "StableSigmoid" ||
        activationName == "Affine" ||
        activationName == "LeakyRelu" || activationName == "LeakyReLU" ||
        activationName == "ThresholdedRelu" || activationName == "ThresholdedReLU" ||
        activationName == "ScaledTanh" ||
        activationName == "HardSigmoid" ||
        activationName == "Elu" || activationName == "ELU" ||
        activationName == "Softsign" ||
        activationName == "Softplus";
}

std::function<FunctionPtr(const Variable &)> ActivationMap(const std::string &activationName)
{
    if (activationName == "Relu")
    {
        return [](const Variable &x) { return ReLU(x); };
    }
    else if (activationName == "Tanh")
    {
        return [](const Variable &x) { return Tanh(x); };
    }
    else if (activationName == "Sigmoid")
    {
        return [](const Variable &x) { return Sigmoid(x); };
    }
    // else if (activationName == "Affine")
    // else if (activationName == "LeakyRelu")
    // else else if (activationName == "ThresholdedRelu")
    // else else if (activationName == "ScaledTanh")
    // else if (activationName == "HardSigmoid")
    else if (activationName == "Elu")
    {
        return [](const Variable &x) { return ELU(x); };
    }
    else if (activationName == "Softsign")
    {
        return [](const Variable &x) { return Softsign(x); };
    }
    else if (activationName == "Softplus")
    {
        return [](const Variable &x) { return Softplus(x); };
    }
    else
    {
        CNTK::LogicError("Recurrent Op does not support activation: %s", activationName.c_str());
    }
}

std::function<FunctionPtr(const Variable &)> ActivationMap(const std::string &activationName,
    float activation_alpha)
{
    if (activationName == "LeakyRelu")
    {
        return [activation_alpha](const Variable &x) { return LeakyReLU(x, activation_alpha); };
    }
    else
    {
        return ActivationMap(activationName);
    }
}

std::function<FunctionPtr(const Variable &)> ActivationMap(const std::string &activationName,
    float activation_alpha, float activation_beta)
{
    if (activationName == "HardSigmoid")
    {
        return [activation_alpha, activation_beta](const Variable &x) { return HardSigmoid(x, activation_alpha, activation_beta); };
    }
    else
    {
        return ActivationMap(activationName, activation_alpha);
    }
}

std::tuple<std::function<FunctionPtr(const Variable &)>, std::function<FunctionPtr(const Variable &)>, std::function<FunctionPtr(const Variable &)>>
GetActivations(const std::vector<std::string> &activations, const std::vector<float> &activation_alpha, const std::vector<float> &activation_beta, int direction)
{
    if (activations.size() < (direction + 1) * LSTMActivationCount)
        CNTK::LogicError("LSTM activations shall be a list of strings of size %d or %d ", LSTMActivationCount, LSTMActivationCount * 2);

    //
    int iofActivationIndex = direction * LSTMActivationCount + LSTMActivationFIndex;
    int cellActivation = direction * LSTMActivationCount + LSTMActivationGIndex;
    int hiddenActivationIndex = direction * LSTMActivationCount + LSTMActivationHIndex;

    // ONNX spec is not clear on how activation alpha and beta is set.
    // Here we assume that if they are set, they are set for all activations, regardless whether
    // an activation needs those values or not.
    bool hasAlpha = activation_alpha.size() == (direction + 1) * LSTMActivationCount;
    bool hasAlphaBeta = hasAlpha && activation_beta.size() == (direction + 1) * LSTMActivationCount;
    std::function<FunctionPtr(const Variable &)> iofActivationOp, cellActivationOp, hiddenActivationOp;
    if (hasAlphaBeta)
    {
        iofActivationOp = ActivationMap(activations[iofActivationIndex], activation_alpha[iofActivationIndex], activation_beta[iofActivationIndex]);
        cellActivationOp = ActivationMap(activations[cellActivation], activation_alpha[cellActivation], activation_beta[cellActivation]);
        hiddenActivationOp = ActivationMap(activations[hiddenActivationIndex], activation_alpha[hiddenActivationIndex], activation_beta[hiddenActivationIndex]);
    }
    else if (hasAlpha)
    {
        iofActivationOp = ActivationMap(activations[iofActivationIndex], activation_alpha[iofActivationIndex]);
        cellActivationOp = ActivationMap(activations[cellActivation], activation_alpha[cellActivation]);
        hiddenActivationOp = ActivationMap(activations[hiddenActivationIndex], activation_alpha[hiddenActivationIndex]);
    }
    else
    {
        iofActivationOp = ActivationMap(activations[iofActivationIndex]);
        cellActivationOp = ActivationMap(activations[cellActivation]);
        hiddenActivationOp = ActivationMap(activations[hiddenActivationIndex]);
    }

    return std::make_tuple(iofActivationOp, cellActivationOp, hiddenActivationOp);
}

std::tuple<std::function<FunctionPtr(const Variable &)>, std::function<FunctionPtr(const Variable &)>>
GetGRUActivations(const std::vector<std::string> &activations, const std::vector<float> &activation_alpha, const std::vector<float> &activation_beta, int direction)
{
    if (activations.size() < (direction + 1) * GRUActivationCount)
        CNTK::LogicError("GRU activations shall be a list of strings of size %d or %d", GRUActivationCount, GRUActivationCount * 2);

    //
    int fActivationIndex = direction * GRUActivationCount + GRUActivationFIndex;
    int gActivationIndex = direction * GRUActivationCount + GRUActivationGIndex;

    bool hasAlpha = activation_alpha.size() == (direction + 1) * GRUActivationCount;
    bool hasAlphaBeta = hasAlpha && activation_beta.size() == (direction + 1) * GRUActivationCount;
    std::function<FunctionPtr(const Variable &)> fActivationOp, gActivationOp;
    if (hasAlphaBeta)
    {
        fActivationOp = ActivationMap(activations[fActivationIndex], activation_alpha[fActivationIndex], activation_beta[fActivationIndex]);
        gActivationOp = ActivationMap(activations[gActivationIndex], activation_alpha[gActivationIndex], activation_beta[gActivationIndex]);
    }
    else if (hasAlpha)
    {
        fActivationOp = ActivationMap(activations[fActivationIndex], activation_alpha[fActivationIndex]);
        gActivationOp = ActivationMap(activations[gActivationIndex], activation_alpha[gActivationIndex]);
    }
    else
    {
        fActivationOp = ActivationMap(activations[fActivationIndex]);
        gActivationOp = ActivationMap(activations[gActivationIndex]);
    }

    return std::make_tuple(fActivationOp, gActivationOp);
}

std::function<FunctionPtr(const Variable &)>
GetRNNActivations(const std::vector<std::string> &activations, const std::vector<float> &activation_alpha, const std::vector<float> &activation_beta, int direction)
{
    if (activations.size() < (direction + 1))
        CNTK::LogicError("RNN activations shall be a list of strings of size 1 or 2");

    //
    int activationIndex = direction;

    bool hasAlpha = activation_alpha.size() == (direction + 1);
    bool hasAlphaBeta = hasAlpha && activation_beta.size() == (direction + 1);
    std::function<FunctionPtr(const Variable &)> activationOp;
    if (hasAlphaBeta)
    {
        activationOp = ActivationMap(activations[activationIndex], activation_alpha[activationIndex], activation_beta[activationIndex]);
    }
    else if (hasAlpha)
    {
        activationOp = ActivationMap(activations[activationIndex], activation_alpha[activationIndex]);
    }
    else
    {
        activationOp = ActivationMap(activations[activationIndex]);
    }

    return activationOp;
}

std::pair<FunctionPtr, FunctionPtr> LSTMPCell(Variable input,
    const std::function<FunctionPtr(const Variable &)> &iofActivationOp,
    const std::function<FunctionPtr(const Variable &)> &cellActivationOp,
    const std::function<FunctionPtr(const Variable &)> &hiddenActivationOp,
    Variable prevOutput, Variable prevCellState,
    Constant &W, Constant &R, Constant &B, Constant &Ci, Constant &Cf, Constant &Co)
{
    size_t outputDim = prevOutput.Shape()[0];
    int stacked_dim = (int)outputDim;

    // computation order shall follow what is in bindings\python\cntk\layers\blocks.py
    // lstm(dh, dc, x)
    FunctionPtr proj4;
    if (B.IsInitialized())
    {
        proj4 = Plus(Plus(B, Times(W, input)), Times(R, prevOutput));
    }
    else
    {
        proj4 = Plus(Times(W, input), Times(R, prevOutput));
    }

    // CNTK weight and bias are in icfo order.
    std::vector<Axis> stack_axis({ Axis(-1) });
    const int IGateIndex = 0, CGateIndex = 1, FGateIndex = 2, OGateIndex = 3;
    FunctionPtr it_proj = Slice(proj4, stack_axis, { IGateIndex * stacked_dim }, { (IGateIndex + 1) * stacked_dim });
    FunctionPtr bit_proj = Slice(proj4, stack_axis, { CGateIndex * stacked_dim }, { (CGateIndex + 1) * stacked_dim });
    FunctionPtr ft_proj = Slice(proj4, stack_axis, { FGateIndex * stacked_dim }, { (FGateIndex + 1) * stacked_dim });
    FunctionPtr ot_proj = Slice(proj4, stack_axis, { OGateIndex * stacked_dim }, { (OGateIndex + 1) * stacked_dim });

    bool hasPeephole = Ci.IsInitialized();

    // Input gate
    auto it = hasPeephole ? iofActivationOp(it_proj + ElementTimes(Ci, prevCellState)) : Sigmoid(it_proj);
    auto bit = ElementTimes(it, cellActivationOp(bit_proj));

    auto ft = hasPeephole ? iofActivationOp(ft_proj + ElementTimes(Cf, prevCellState)) : Sigmoid(ft_proj);
    auto bft = ElementTimes(ft, prevCellState);

    auto ct = Plus(bft, bit);

    auto ot = hasPeephole ? iofActivationOp(ot_proj + ElementTimes(Co, ct)) : Sigmoid(ot_proj);
    auto ht = ElementTimes(ot, hiddenActivationOp(ct));

    auto c = ct;
    auto h = ht;

    return { h, c };
}

FunctionPtr GRUCell(Variable input,
    const std::function<FunctionPtr(const Variable &)> &fActivationOp,
    const std::function<FunctionPtr(const Variable &)> &gActivationOp,
    Variable prevOutput,
    Constant &W, Constant &R, Constant &H1, Constant &B)
{
    size_t outputDim = prevOutput.Shape()[0];
    int stacked_dim = (int)outputDim;

    // computation order shall follow what is in bindings\python\cntk\layers\blocks.py
    // gru(dh, x)

    FunctionPtr projx3;
    if (B.IsInitialized())
        projx3 = Plus(B, Times(W, input));
    else
        projx3 = Times(W, input);

    FunctionPtr projh2 = Times(R, prevOutput);

    // both CNTK and ONNX weight and bias are in zrh order.
    std::vector<Axis> stack_axis({ Axis(-1) });
    FunctionPtr zt_proj =
        Slice(projx3, stack_axis, { 0 * stacked_dim }, { 1 * stacked_dim }) +
        Slice(projh2, stack_axis, { 0 * stacked_dim }, { 1 * stacked_dim });

    FunctionPtr rt_proj =
        Slice(projx3, stack_axis, { 1 * stacked_dim }, { 2 * stacked_dim }) +
        Slice(projh2, stack_axis, { 1 * stacked_dim }, { 2 * stacked_dim });

    FunctionPtr ct_proj =
        Slice(projx3, stack_axis, { 2 * stacked_dim }, { 3 * stacked_dim });

    FunctionPtr zt = fActivationOp(zt_proj);

    FunctionPtr rt = fActivationOp(rt_proj);

    FunctionPtr rs = ElementTimes(prevOutput, rt);

    FunctionPtr ct = gActivationOp(ct_proj + Times(H1, rs));

    Constant one = Constant::Scalar(W.GetDataType(), 1.0);

    FunctionPtr ht = ElementTimes(one - zt, ct) + ElementTimes(zt, prevOutput);

    FunctionPtr h = ht;

    return ht;
}

FunctionPtr RNNCell(Variable input,
    const std::function<FunctionPtr(const Variable &)> &activationOp,
    Variable prevOutput,
    Constant &W, Constant &R, Constant &B)
{
    // computation order shall follow what is in bindings\python\cntk\layers\blocks.py
    // rnn_step(dh, x)
    FunctionPtr proj = Times(W, input) + Times(R, prevOutput);

    if (B.IsInitialized())
        proj = proj + B;

    FunctionPtr h = activationOp(proj);
    return h;
}

#include "PrimitiveFunction.h"
#include "PrimitiveFunctionAttribute.h"
#include "BlockFunction.h"

std::tuple<FunctionPtr, FunctionPtr> LSTMPComponent(Variable input,
    const NDShape &cellShape,
    const std::function<FunctionPtr(const Variable &)> &iofActivationOp,
    const std::function<FunctionPtr(const Variable &)> &cellActivationOp,
    const std::function<FunctionPtr(const Variable &)> &hiddenActivationOp,
    const std::function<FunctionPtr(const Variable &)> &recurrenceHookH,
    const std::function<FunctionPtr(const Variable &)> &recurrenceHookC,
    Constant &W, Constant &R, Constant &B,
    Constant &Ci, Constant &Cf, Constant &Co)
{
    auto dh = PlaceholderVariable(cellShape, input.DynamicAxes());
    auto dc = PlaceholderVariable(cellShape, input.DynamicAxes());
    auto dh2 = PlaceholderVariable(cellShape, input.DynamicAxes());
    auto dc2 = PlaceholderVariable(cellShape, input.DynamicAxes());
    auto inputPlaceholder = PlaceholderVariable(input.Shape(), input.DynamicAxes());

    auto LSTMCell = LSTMPCell(
        inputPlaceholder,
        iofActivationOp, cellActivationOp, hiddenActivationOp,
        dh, dc, W, R, B, Ci, Cf, Co);

    auto actualDh = recurrenceHookH(dh2);
    auto actualDc = recurrenceHookC(dc2);

    auto LSTMCellcombined = Combine({ LSTMCell.first, LSTMCell.second });

    auto LSTMCellcombinedBlk = AsBlock(std::move(LSTMCellcombined), { { inputPlaceholder, input },{ dh, actualDh },{ dc, actualDc } }, L"LSTM", L"");

    actualDh->ReplacePlaceholders({ { dh2, LSTMCellcombinedBlk->Outputs()[0] } });
    actualDc->ReplacePlaceholders({ { dc2, LSTMCellcombinedBlk->Outputs()[1] } });

    // Because state and cell variables share the same owner function, we need
    // to use Alias so that they can be differenciated when building subsequent graph.
    return std::make_tuple(Alias(LSTMCellcombinedBlk->Outputs()[0]),
        Alias(LSTMCellcombinedBlk->Outputs()[1]));
}
FunctionPtr GRUComponent(Variable input,
    const NDShape &cellShape,
    const std::function<FunctionPtr(const Variable &)> &fActivationOp,
    const std::function<FunctionPtr(const Variable &)> &gActivationOp,
    const std::function<FunctionPtr(const Variable &)> &recurrenceHookH,
    Constant &W, Constant &R, Constant &H1, Constant &B)
{
    auto dh = PlaceholderVariable(cellShape, input.DynamicAxes());
    auto dh2 = PlaceholderVariable(cellShape, input.DynamicAxes());
    auto inputPlaceholder = PlaceholderVariable(input.Shape(), input.DynamicAxes());

    auto gruCell = GRUCell(
        inputPlaceholder,
        fActivationOp, gActivationOp,
        dh, W, R, H1, B);

    auto actualDh = recurrenceHookH(dh2);
    auto gruBlock = AsBlock(std::move(gruCell), { { inputPlaceholder, input },{ dh, actualDh } }, L"GRU", L"");
    actualDh->ReplacePlaceholders({ { dh2, gruBlock } });
    return gruBlock;
}

FunctionPtr RNNComponent(Variable input,
    const NDShape &cellShape,
    const std::function<FunctionPtr(const Variable &)> &activationOp,
    const std::function<FunctionPtr(const Variable &)> &recurrenceHookH,
    Constant &W, Constant &R, Constant &B)
{
    auto dh = PlaceholderVariable(cellShape, input.DynamicAxes());
    auto dh2 = PlaceholderVariable(cellShape, input.DynamicAxes());
    auto inputPlaceholder = PlaceholderVariable(input.Shape(), input.DynamicAxes());

    auto rnnCell = RNNCell(
        inputPlaceholder,
        activationOp,
        dh, W, R, B);

    auto actualDh = recurrenceHookH(dh2);
    auto rnnBlock = AsBlock(std::move(rnnCell), { { inputPlaceholder, input },{ dh, actualDh } }, L"RNNStep", L"");
    actualDh->ReplacePlaceholders({ { dh2, rnnBlock } });
    return rnnBlock;
}

const std::vector<Variable> FindByNameHint(const std::vector<Variable> &inputs, const std::string &hint)
{
    std::vector<Variable> variables;
    for (auto v : inputs)
    {
        if (ToLegacyString(ToUTF8(v.Name())).find(hint) != -1)
        {
            variables.push_back(v);
        }
    }
    return variables;
}

Variable GetInitialStateVariable(const std::vector<Variable> &inputs, int numDirections,
    const std::string &nameHint, CNTK::DataType datatype)
{
    Variable initialVariable = datatype == CNTK::DataType::Double ? Constant::Scalar(0.0) : Constant::Scalar(0.0f);
    const std::vector<Variable> initialVariables = FindByNameHint(inputs, nameHint);
    if (numDirections == 1 && initialVariables.size() >= 1)
    {
        initialVariable = initialVariables[0];
    }
    else if (numDirections == 2 && initialVariables.size() >= 2)
    {
        initialVariable = initialVariables[1];
    }

    return initialVariable;
}

// sequenceWrapperInputToFunctionPtr is used when more than one RNN op takes one same variable as the input.
// in this case, we only want to wrap RNN ops with the input once.
Variable ToBatchAndSequence(Variable input, VariableToFunctionPtr &sequenceWrapperInputToFunctionPtr)
{
    if (sequenceWrapperInputToFunctionPtr.find(input) != sequenceWrapperInputToFunctionPtr.end())
        return sequenceWrapperInputToFunctionPtr[input];

    if (input.DynamicAxes().size() != 0)
        return input;

    if(input.DynamicAxes().size() != 0)
        CNTK::LogicError("Input (%s) shall not have any dynamic axis", ToLegacyString(ToUTF8(input.Name())).c_str());
    if (input.Shape().Rank() < 2)
        CNTK::LogicError("Shape of input (%s) shall have rank that is equal or more than 2", ToLegacyString(ToUTF8(input.Name())).c_str());

    FunctionPtr transpose = TransposeAxes(input, Axis(input.Shape().Rank() - 2), Axis(input.Shape().Rank() - 1), L"");
    FunctionPtr toBatch = ToBatch(transpose, L"wrapper_sequence_axis");
    FunctionPtr operandWithBatchAndSequenceAxis = ToSequence(toBatch, L"ONNX_export_RNN_wrapper_sequence_axis", L"");
    sequenceWrapperInputToFunctionPtr[input] = operandWithBatchAndSequenceAxis;
    return operandWithBatchAndSequenceAxis;
}

FunctionPtr UnpackBatchAndSequence(FunctionPtr rnnFunction, bool doTranspose)
{
    FunctionPtr cntkFunctionWithoutSequenceAxis = Sequence::Unpack(rnnFunction, 0, L"");
    FunctionPtr cntkFunctionWithoutDynamicAxis = UnpackBatch(cntkFunctionWithoutSequenceAxis, L"");
    if (doTranspose)
    {
        FunctionPtr transpose = TransposeAxes(cntkFunctionWithoutDynamicAxis,
            Axis(cntkFunctionWithoutDynamicAxis->Output().Shape().Rank() - 2),
            Axis(cntkFunctionWithoutDynamicAxis->Output().Shape().Rank() - 1), L"");
        return transpose;
    }
    else
        // in case of RNN ops, transpose is inserted after the op so we do not do transpose again
        // TODO: do not transpose after RNN ops so we have one code path here.
        return cntkFunctionWithoutDynamicAxis;
}


FunctionPtr UnwrapRNNOps(FunctionPtr rnnFunction, int numDirections)
{
    // [#, *][dirs * hidden] 
    FunctionPtr cntkFunctionWithoutSequenceAxis = Sequence::Unpack(rnnFunction, 0, L"");
    // [#][*, dirs * hidden] 
    FunctionPtr cntkFunctionWithoutDynamicAxis = UnpackBatch(cntkFunctionWithoutSequenceAxis, L"");
    // [#, *, dirs * hidden] 

    NDShape newShape = cntkFunctionWithoutDynamicAxis->Output().Shape();
    int hidden = newShape[0] / numDirections;
    newShape = newShape.AppendShape({ NDShape::FreeDimension });
    newShape[2] = numDirections;
    newShape[1] = FreeBatchSize;
    newShape[0] = hidden;
    // because FreeBatchSize = 1, we can skip transpose between # and dirs.
    FunctionPtr cntkFunctionWithoutDynamicAxisFixedBatch = Reshape(cntkFunctionWithoutDynamicAxis, newShape, L"");
    // [*, dirs, #, hidden]
    return cntkFunctionWithoutDynamicAxisFixedBatch;
}

FunctionPtr CreateLSTM(const onnxruntime::Node *node, const std::vector<Variable> &inputs, const std::string &direction,
    const std::vector<string> &activations, const std::vector<float> &activation_alpha, const std::vector<float> &activation_beta,
    VariableToFunctionPtr &sequenceWrapperInputToFunctionPtr)
{
    int numDirections = direction == RNNDirectionBidirection ? 2 : 1;
    std::vector<FunctionPtr> outputHs;
    Variable X;
    X = ToBatchAndSequence(inputs[0], sequenceWrapperInputToFunctionPtr);
    for (int dir = 0; dir < numDirections; dir++)
    {
        std::function<FunctionPtr(const Variable &)> iofActivationOp, cellActivationOp, hiddenActivationOp;
        std::tie<std::function<FunctionPtr(const Variable &)>, std::function<FunctionPtr(const Variable &)>, std::function<FunctionPtr(const Variable &)>>(iofActivationOp, cellActivationOp, hiddenActivationOp) = GetActivations(activations, activation_alpha, activation_beta, dir);

        // the first a few inputs are (in order): X, numDirections * W, numDirections * R
        Variable W = inputs[1 + dir];
        Variable R = inputs[1 + numDirections + dir];
        Variable B;
        std::vector<Variable> biasVariables = FindByNameHint(inputs, LSTMInputBiasNameHint);
        if (numDirections == 1 && biasVariables.size() >= 1)
            B = biasVariables[dir];
        else if (numDirections == 2 && biasVariables.size() == 2)
            B = biasVariables[dir];

        Variable initHVariable = GetInitialStateVariable(inputs, numDirections, LSTMInputInitialHNameHint, X.GetDataType());
        Variable initCVariable = GetInitialStateVariable(inputs, numDirections, LSTMInputInitialCNameHint, X.GetDataType());

        std::vector<Variable> peepholeVariables = FindByNameHint(inputs, LSTMInputPeepholeNameHint);
        Variable Ci, Cf, Co;
        if (peepholeVariables.size() != 0 && peepholeVariables.size() != LSTMPeepholeCount && peepholeVariables.size() != 2 * LSTMPeepholeCount)
        {
            CNTK::LogicError("Peephole Variable count (%d) should be 0, 1 or 2 times the number of peephole factors (%d).",
                (int)(peepholeVariables.size()), (int)LSTMPeepholeCount);
        }
        else if (numDirections == 1 && peepholeVariables.size() >= LSTMPeepholeCount)
        {
            Ci = peepholeVariables[LSTMPeepholeCountCiIndex];
            Co = peepholeVariables[LSTMPeepholeCountCoIndex];
            Cf = peepholeVariables[LSTMPeepholeCountCfIndex];
        }
        else if (numDirections == 2 && peepholeVariables.size() == numDirections * LSTMPeepholeCount)
        {
            Ci = peepholeVariables[LSTMPeepholeCount + LSTMPeepholeCountCiIndex];
            Co = peepholeVariables[LSTMPeepholeCount + LSTMPeepholeCountCoIndex];
            Cf = peepholeVariables[LSTMPeepholeCount + LSTMPeepholeCountCfIndex];
        }

        // ONNX spec https://github.com/onnx/onnx/blob/master/docs/Operators.md#inputs-3---8
        // tells that weight has shape [num_directions, 4*hidden_size, input_size]
        // here in CNTK, there is no direction axis because CNTK treats bidirectional LSTM
        // as two separate LSTM. Therefore we can divide the dimension of the first axis
        // by 4 to get the hidden size.
        int hiddenDim = W.Shape()[0] / 4;

        FunctionPtr outputH;
        FunctionPtr outputC;

        // if it is bidirectional LSTM, the second one will be the backword one.
        bool go_backwards = direction == RNNDirectionReverse || (numDirections == 2 && dir == 1);

        std::function<FunctionPtr(const Variable &)> recurrenceHookH, recurrenceHookC;
        if (go_backwards)
        {
            recurrenceHookH = [initHVariable](const Variable &x) { return FutureValue(x, initHVariable); };
            recurrenceHookC = [initCVariable](const Variable &x) { return FutureValue(x, initCVariable); };
        }
        else
        {
            recurrenceHookH = [initHVariable](const Variable &x) { return PastValue(x, initHVariable); };
            recurrenceHookC = [initCVariable](const Variable &x) { return PastValue(x, initCVariable); };
        }

        std::tie<FunctionPtr, FunctionPtr>(outputH, outputC) = LSTMPComponent(
            X, { (size_t)hiddenDim }, iofActivationOp, cellActivationOp, hiddenActivationOp,
            recurrenceHookH, recurrenceHookC, (Constant &)W, (Constant &)R, (Constant &)B,
            (Constant &)Ci, (Constant &)Cf, (Constant &)Co);

        outputHs.push_back(outputH);
    }

    FunctionPtr rnnFunction;
    if (outputHs.size() == 1)
        rnnFunction = outputHs[0];
    else
    {
        std::vector<Variable> operands({ outputHs[0], outputHs[1] });
        rnnFunction = Splice(operands, Axis(0), ToFixedWStringFromMultiByte(node->Name()));
    }

    FunctionPtr unpackedRnnFunction = UnwrapRNNOps(rnnFunction, outputHs.size());
    return unpackedRnnFunction;
}

FunctionPtr CreateGRU(const onnxruntime::Node *node, const std::vector<Variable> &inputs, const std::string &direction,
    const std::vector<string> &activations, const std::vector<float> &activation_alpha, const std::vector<float> &activation_beta,
    VariableToFunctionPtr &sequenceWrapperInputToFunctionPtr)
{
    int numDirections = direction == RNNDirectionBidirection ? 2 : 1;
    std::vector<FunctionPtr> outputHs;
    Variable X = ToBatchAndSequence(inputs[0], sequenceWrapperInputToFunctionPtr);

    for (int dir = 0; dir < numDirections; dir++)
    {
        std::function<FunctionPtr(const Variable &)> fActivationOp, gActivationOp;
        std::tie<std::function<FunctionPtr(const Variable &)>, std::function<FunctionPtr(const Variable &)>>(fActivationOp, gActivationOp) = GetGRUActivations(activations, activation_alpha, activation_beta, dir);

        // the first a few inputs are (in order): X, numDirections * W, numDirections * R, numDirections * H1
        Variable W = inputs[1 * numDirections + dir - ((numDirections == 2) ? 1 : 0)];
        Variable R = inputs[2 * numDirections + dir - ((numDirections == 2) ? 1 : 0)];

        // TODO: get H1
        Variable H1 = inputs[3 * numDirections + dir - ((numDirections == 2) ? 1 : 0)];

        Variable B;
        std::vector<Variable> biasVariables = FindByNameHint(inputs, LSTMInputBiasNameHint);
        if (numDirections == 1 && biasVariables.size() >= 1)
            B = biasVariables[dir];
        else if (numDirections == 2 && biasVariables.size() == 2)
            B = biasVariables[dir];

        Variable initHVariable = GetInitialStateVariable(inputs, numDirections, GRUInputInitialHNameHint, X.GetDataType());

        // ONNX spec https://github.com/onnx/onnx/blob/master/docs/Operators.md#inputs-3---8
        // tells that weight has shape [num_directions, 4*hidden_size, input_size]
        // here in CNTK, there is no direction axis because CNTK treats bidirectional LSTM
        // as two separate LSTM. Therefore we can divide the dimension of the first axis
        // by 4 to get the hidden size.
        int hiddenDim = W.Shape()[0] / GRUWeightDimensionHiddenMultiplier;

        FunctionPtr outputH;

        // if it is bidirectional LSTM, the second one will be the backword one.
        bool go_backwards = direction == RNNDirectionReverse || (numDirections == 2 && dir == 1);

        std::function<FunctionPtr(const Variable &)> recurrenceHook;
        if (go_backwards)
            recurrenceHook = [initHVariable](const Variable &x) { return FutureValue(x, initHVariable); };
        else
            recurrenceHook = [initHVariable](const Variable &x) { return PastValue(x, initHVariable); };

        outputH = GRUComponent(
            X, { (size_t)hiddenDim }, fActivationOp, gActivationOp,
            recurrenceHook, (Constant &)W, (Constant &)R, (Constant &)H1, (Constant &)B);
        outputHs.push_back(outputH);
    }

    FunctionPtr rnnFunction;
    if (outputHs.size() == 1)
        rnnFunction = outputHs[0];
    else
    {
        std::vector<Variable> operands({ outputHs[0], outputHs[1] });
        rnnFunction = Splice(operands, Axis(0), ToFixedWStringFromMultiByte(node->Name()));
    }

    FunctionPtr unpackedRnnFunction = UnwrapRNNOps(rnnFunction, outputHs.size());
    return unpackedRnnFunction;
}

FunctionPtr CreateRNN(const onnxruntime::Node *node, const std::vector<Variable> &inputs, const std::string &direction,
    const std::vector<string> &activations, const std::vector<float> &activation_alpha, const std::vector<float> &activation_beta,
    VariableToFunctionPtr &sequenceWrapperInputToFunctionPtr)
{
    int numDirections = direction == RNNDirectionBidirection ? 2 : 1;
    std::vector<FunctionPtr> outputHs;
    Variable X = ToBatchAndSequence(inputs[0], sequenceWrapperInputToFunctionPtr);

    for (int dir = 0; dir < numDirections; dir++)
    {
        std::function<FunctionPtr(const Variable &)> activationOp =
            GetRNNActivations(activations, activation_alpha, activation_beta, dir);

        // the first a few inputs are (in order): X, numDirections * W, numDirections * R, numDirections * H1
        Variable W = inputs[1 * numDirections + dir - ((numDirections == 2) ? 1 : 0)];
        Variable R = inputs[2 * numDirections + dir - ((numDirections == 2) ? 1 : 0)];
        Variable B;
        std::vector<Variable> biasVariables = FindByNameHint(inputs, LSTMInputBiasNameHint);
        if (numDirections == 1 && biasVariables.size() >= 1)
            B = biasVariables[dir];
        else if (numDirections == 2 && biasVariables.size() == 2)
            B = biasVariables[dir];

        Variable initHVariable = GetInitialStateVariable(inputs, numDirections, GRUInputInitialHNameHint, X.GetDataType());

        int hiddenDim = W.Shape()[0];

        FunctionPtr outputH;

        // if it is bidirectional LSTM, the second one will be the backword one.
        bool go_backwards = direction == RNNDirectionReverse || (numDirections == 2 && dir == 1);

        std::function<FunctionPtr(const Variable &)> recurrenceHook;
        if (go_backwards)
            recurrenceHook = [initHVariable](const Variable &x) { return FutureValue(x, initHVariable); };
        else
            recurrenceHook = [initHVariable](const Variable &x) { return PastValue(x, initHVariable); };

        outputH = RNNComponent(
            X, { (size_t)hiddenDim }, activationOp,
            recurrenceHook, (Constant &)W, (Constant &)R, (Constant &)B);
        outputHs.push_back(outputH);
    }

    FunctionPtr rnnFunction;
    if (outputHs.size() == 1)
        rnnFunction = outputHs[0];
    else
    {
        std::vector<Variable> operands({ outputHs[0], outputHs[1] });
        rnnFunction = Splice(operands, Axis(0), ToFixedWStringFromMultiByte(node->Name()));
    }

    FunctionPtr unpackedRnnFunction = UnwrapRNNOps(rnnFunction, outputHs.size());
    return unpackedRnnFunction;
}

template <typename FunctionType>
void TraverseGraphWithPrePostActions(FunctionPtr cntkFunction, std::unordered_set<FunctionPtr> &visitedFunctions,
    FunctionType preFunctor, FunctionType postFunctor)
{
    visitedFunctions.insert(cntkFunction);
    preFunctor(cntkFunction);

    std::vector<Variable> functionInputs = cntkFunction->Inputs();
    for (const auto &input : functionInputs)
    {
        if (input.IsOutput() && visitedFunctions.find(input.Owner()) == visitedFunctions.end())
        {
            const auto &inputFunction = input.Owner();
            TraverseGraphWithPrePostActions(inputFunction, visitedFunctions, preFunctor, postFunctor);
        }
    }

    postFunctor(cntkFunction);
}

bool IsSupportedRNNActivation(const std::wstring &cntkOpName)
{
    static std::vector<std::wstring> supportedRNNActivations(
        { L"ReLU",
        L"Tanh",
        L"StableSigmoid" });
    return std::find(supportedRNNActivations.cbegin(), supportedRNNActivations.cend(), cntkOpName) !=
        supportedRNNActivations.cend();
}

std::string FindActivation(const std::vector<FunctionPtr> &path, int nth)
{
    int count = 0;
    for (std::vector<FunctionPtr>::const_iterator it = path.begin(); it != path.end(); it++)
    {
        std::wstring opName = (*it)->OpName();
        if (IsSupportedRNNActivation(opName))
        {
            if (count == nth)
            {
                std::unordered_multimap<std::wstring, AttributesMapping>::const_iterator itLookup = Operators::CntkToONNXLookup().find(opName);
                if (itLookup == Operators::CntkToONNXLookup().cend())
                    CNTK::LogicError("Invalid activation (%s)", ToLegacyString(ToUTF8(opName)).c_str());

                std::unordered_map<std::wstring, std::string>::const_iterator itMap = (*itLookup).second.map.find(opName);
                if (itMap == (*itLookup).second.map.cend())
                    CNTK::LogicError("Invalid activation (%s)", ToLegacyString(ToUTF8(opName)).c_str());
                return itMap->second;
            }
            count++;
        }
    }
    return "";
}

Variable GetPeepholeVariableFromOp(FunctionPtr peepholeOp)
{
    // peephole variable is that child of peepholeOp that is neither stabilizer nor place holder
    if (peepholeOp->OpName() != L"ElementTimes")
        CNTK::LogicError("Peephole operation must be ElementTimes");

    Variable peepholeVariable;
    FunctionPtr stabilizerOp;
    for (int i = 0; i < peepholeOp->Inputs().size(); i++)
    {
        if (peepholeOp->Inputs()[i].Owner() && peepholeOp->Inputs()[i].Owner()->OpName() == L"Stabilizer")
        {
            stabilizerOp = peepholeOp->Inputs()[i].Owner();
        }
        else if (peepholeOp->Inputs()[i].IsConstant() || peepholeOp->Inputs()[i].IsParameter())
        {
            if (!peepholeVariable.IsInitialized())
                peepholeVariable = peepholeOp->Inputs()[i];
            else
                CNTK::LogicError("Cannot find peephole variable from peephole op. Multiple qualified variables found.");
        }
    }

    if (!peepholeVariable.IsInitialized())
        CNTK::LogicError("Cannot find peephole variable from peephole op.");
    return peepholeVariable;
}

// this method helps getting a stabilizer op from its parent/grandparent op.
// the parent op can be Times or ElementTimes. The grandparent op can be a plus op.
FunctionPtr GetStabilizerOp(FunctionPtr parentOp)
{
    FunctionPtr timesOp;
    if (parentOp->OpName() == L"Plus")
    {
        for (int i = 0; i < parentOp->Inputs().size(); i++)
        {
            if (parentOp->Inputs()[i].Owner() &&
                (parentOp->Inputs()[i].Owner()->OpName() == L"Times" ||
                    parentOp->Inputs()[i].Owner()->OpName() == L"ElementTimes"))
            {
                timesOp = parentOp->Inputs()[i].Owner();
                break;
            }
        }
    }
    else if (parentOp->OpName() == L"Times" || parentOp->OpName() == L"ElementTimes")
    {
        timesOp = parentOp;
    }

    if (!timesOp)
    {
        CNTK::LogicError("Cannot find stabilizer op. A stabilizer op must be from Times or ElementTimes ops or skipped from a Plus op.");
    }

    for (int j = 0; j < timesOp->Inputs().size(); j++)
    {
        if (timesOp->Inputs()[j].Owner() && timesOp->Inputs()[j].Owner()->OpName() == L"Stabilizer")
        {
            return timesOp->Inputs()[j].Owner();
        }
    }

    return nullptr;
}

double GetScaler(Variable variable)
{
    NDArrayViewPtr v = variable.IsParameter() ? Parameter(variable).Value() : Constant(variable).Value();
    NDArrayViewPtr cpuV = v->DeepClone();
    cpuV->ChangeDevice(DeviceDescriptor::CPUDevice());

    switch (variable.GetDataType())
    {
    case CNTK::DataType::Float:
        return *((float *)cpuV->DataBuffer<float>());
    case CNTK::DataType::Double:
        return *((double *)cpuV->DataBuffer<double>());
    default:
        NOT_IMPLEMENTED;
    }
}

double GetStabilizerCoef(const FunctionPtr stabilizerDhOp)
{
    double alpha = GetScaler(stabilizerDhOp->Inputs()[3]);
    double steepness = GetScaler(stabilizerDhOp->Inputs()[1]);
    return (log(exp(alpha * steepness) + 1.0F) / steepness);
}

void GetDelayOps(const std::vector<Variable> &inputVars,
    std::vector<FunctionPtr> &pastValueOps, std::vector<FunctionPtr> &futureValueOps)
{
    for (std::vector<Variable>::const_iterator it = inputVars.cbegin(); it != inputVars.cend(); ++it)
    {
        if ((*it).Owner() != nullptr && (*it).Owner()->OpName() == L"PastValue")
            pastValueOps.push_back((*it).Owner());
        else if ((*it).Owner() != nullptr && (*it).Owner()->OpName() == L"FutureValue")
            futureValueOps.push_back((*it).Owner());
    }
}

// A CNTK LSTM op is created with stacked matmul followed by a slice op for 4 gates.
// Slice order tells which graph path is for which gate. This method is
// to traverse the graph to find the 4 paths along the 4 gates. It helps to
// subsequently find needed attributes in order to build an ONNX LSTM op.
void TraceLSTMPathes(const FunctionPtr &src,
    string &f_activation,
    string &g_activation,
    string &h_activation,
    RNNDirection &direction,
    Variable &initStateH,
    Variable &initStateC,
    Variable &peepholeCi,
    Variable &peepholeCo,
    Variable &peepholeCf,
    double &stabilizer_dh,
    double &stabilizer_dc,
    double &stabilizer_c)
{
    // src has to be an LSTM node.
    std::vector<Variable> inputVars = src->Inputs();
    std::vector<FunctionPtr> pastValueOps, futureValueOps;
    GetDelayOps(inputVars, pastValueOps, futureValueOps);

    // with CNTK LSTM, the first delay node is for H, the second one is for C
    // indices here also coresponding with CNTK python layer code.
    if (pastValueOps.size() == 2 && futureValueOps.size() == 0)
    {
        direction = RNNDirection::Forward;
        initStateH = pastValueOps[0]->Inputs()[1];
        initStateC = pastValueOps[1]->Inputs()[1];
    }
    else if (pastValueOps.size() == 0 && futureValueOps.size() == 2)
    {
        direction = RNNDirection::Backward;
        initStateH = futureValueOps[0]->Inputs()[1];
        initStateC = futureValueOps[1]->Inputs()[1];
    }
    else
    {
        CNTK::LogicError("Node %s (%s) is not a valid LSTM node", ToLegacyString(ToUTF8(src->Name())).c_str(), ToLegacyString(ToUTF8(src->Uid())).c_str());
    }

    // set up traverse boundary
    std::unordered_set<FunctionPtr> visitedFunctions;
    for (std::vector<Variable>::const_iterator it = inputVars.begin(); it != inputVars.end(); it++)
    {
        visitedFunctions.insert(it->Owner());
    }

    // First find the peephole op node.
    // see CNTK\bindings\python\cntk\layers\blocks.py node references.
    std::vector<std::vector<FunctionPtr>> pathesBitBftJoint;
    {
        std::vector<FunctionPtr> currentPeepholePath;

        // make a copy of traverse boundary
        std::unordered_set<FunctionPtr> peepHoleVisitedFunctions = visitedFunctions;

        // traverse to find the joint of bit and bft
        TraverseGraphWithPrePostActions(src->BlockRoot(),
            peepHoleVisitedFunctions,
            (std::function<void(const FunctionPtr &)>) [
                &peepHoleVisitedFunctions, &pathesBitBftJoint, &currentPeepholePath
            ](const FunctionPtr &function) {
            currentPeepholePath.push_back(function);
            if (function->OpName() == L"Plus" &&
                function->Inputs()[0].Owner() && function->Inputs()[0].Owner()->OpName() == L"ElementTimes" &&
                function->Inputs()[1].Owner() && function->Inputs()[1].Owner()->OpName() == L"ElementTimes")
            {
                pathesBitBftJoint.push_back(currentPeepholePath);
                peepHoleVisitedFunctions.erase(std::find_if(peepHoleVisitedFunctions.begin(), peepHoleVisitedFunctions.end(),
                    [function](FunctionPtr f) { return function == f; }));
            }
        },
                    (std::function<void(const FunctionPtr &)>) [&currentPeepholePath](const FunctionPtr &function) {
            currentPeepholePath.pop_back();
        });
    }

    FunctionPtr peepholeCoOp;
    bool haspeephole = pathesBitBftJoint.size() == 3;
    if (haspeephole)
    {
        // the last ElementTimes op is the peephole op
        std::vector<FunctionPtr> &peepholePath = *std::max_element(pathesBitBftJoint.begin(), pathesBitBftJoint.end(),
            [](std::vector<FunctionPtr> &p1, std::vector<FunctionPtr> &p2) { return p1.size() < p2.size(); });
        std::vector<FunctionPtr>::reverse_iterator itPeepholeOp = std::find_if(peepholePath.rbegin(), peepholePath.rend(),
            [](FunctionPtr function) { return function->OpName() == L"ElementTimes"; });
        if (itPeepholeOp == peepholePath.rend())
        {
            CNTK::LogicError("Cannot find peephole op from a LSTM graph");
        }

        peepholeCoOp = *itPeepholeOp;
        peepholeCo = GetPeepholeVariableFromOp(peepholeCoOp);

        FunctionPtr stabilizer_h_op = GetStabilizerOp(peepholeCoOp);
        if (stabilizer_h_op)
        {
            stabilizer_c = GetStabilizerCoef(stabilizer_h_op);
        }
    }

    std::vector<std::vector<FunctionPtr>> pathesToPlusSlice;
    std::vector<FunctionPtr> currentPath;

    if (haspeephole)
        // so that traverse will not be affected by the peephole path
        visitedFunctions.insert(peepholeCoOp);

    TraverseGraphWithPrePostActions(src->BlockRoot(),
        visitedFunctions,
        (std::function<void(const FunctionPtr &)>) [&pathesToPlusSlice, &currentPath](const FunctionPtr &function) {
        currentPath.push_back(function);
        if (function->OpName() == L"Slice")
        {
            FunctionPtr functionSource = function->Inputs()[0].Owner();
            if (functionSource->OpName() == L"Plus")
            {
                pathesToPlusSlice.push_back(currentPath);
            }
        }
    },
        (std::function<void(const FunctionPtr &)>) [&currentPath](const FunctionPtr &function) {
        currentPath.pop_back();
    });

    // 4 gates of LSTM shall be traced.
    if (pathesToPlusSlice.size() != 4)
    {
        CNTK::LogicError("pathesToPlusSlice.size() != 4");
    }

    std::sort(pathesToPlusSlice.begin(), pathesToPlusSlice.end(),
        [](const std::vector<FunctionPtr> &path1, const std::vector<FunctionPtr> &path2) {
        FunctionPtr slice1 = *path1.rbegin();
        FunctionPtr slice2 = *path2.rbegin();
        int beginIndex1 = slice1->Attributes()[PrimitiveFunctionAttribute::AttributeNameBeginIndex].Value<int>();
        int beginIndex2 = slice2->Attributes()[PrimitiveFunctionAttribute::AttributeNameBeginIndex].Value<int>();
        return beginIndex1 < beginIndex2;
    });

    // This code is heavily coupled with CNTK python layer code:
    // https://github.com/Microsoft/CNTK/blob/44c626a483edeaff97b4f7a46847b055a1d483aa/bindings/python/cntk/layers/blocks.py#L261
    // pathesToPlusSlice is ordered by slice index so we are able to recover corresponding path here.
    std::vector<FunctionPtr> &ht_it_path = pathesToPlusSlice[0];
    std::vector<FunctionPtr> &ht_bit_path = pathesToPlusSlice[1];
    std::vector<FunctionPtr> &ht_ft_path = pathesToPlusSlice[2];
    std::vector<FunctionPtr> &ht_ot_path = pathesToPlusSlice[3];

    f_activation = MapActivationNameCNTKToONNX(FindActivation(ht_ot_path, 0));
    g_activation = MapActivationNameCNTKToONNX(FindActivation(ht_bit_path, 1));
    h_activation = MapActivationNameCNTKToONNX(FindActivation(ht_bit_path, 0));

    // stabilizer_dh
    FunctionPtr stackedProjPlusOp = ht_it_path[ht_it_path.size() - 1]->Inputs()[0].Owner();
    FunctionPtr stabilizerDhOp = GetStabilizerOp(stackedProjPlusOp);
    if (stabilizerDhOp)
    {
        stabilizer_dh = GetStabilizerCoef(stabilizerDhOp);
    }

    if (haspeephole)
    {
        {
            // Ci merges to ht_it_path via element-wise time
            FunctionPtr plusOp = ht_it_path[ht_it_path.size() - 2];
            FunctionPtr peepholeOp = plusOp->Inputs()[0].Owner()->OpName() != L"Slice" ? plusOp->Inputs()[0].Owner() : plusOp->Inputs()[1].Owner();
            peepholeCi = GetPeepholeVariableFromOp(peepholeOp);
        }
        {
            // Cf merges to ht_ft_path via element-wise time
            FunctionPtr plusOp = ht_ft_path[ht_ft_path.size() - 2];
            FunctionPtr peepholeOp = plusOp->Inputs()[0].Owner()->OpName() != L"Slice" ? plusOp->Inputs()[0].Owner() : plusOp->Inputs()[1].Owner();
            peepholeCf = GetPeepholeVariableFromOp(peepholeOp);

            FunctionPtr stabilizerDcOp = GetStabilizerOp(peepholeOp);
            if (stabilizerDcOp)
                stabilizer_dc = GetStabilizerCoef(stabilizerDcOp);
        }
    }
}

FunctionPtr TraverseGraphFindFirstRNNOp(FunctionPtr src)
{
    std::vector<Variable> front = src->Inputs(), back;

    while (!front.empty())
    {
        for (auto f : front)
        {
            if (f.IsOutput() && f.Owner())
                if (IsActivationOp(ToLegacyString(ToUTF8(f.Owner()->OpName()))))
                    return f.Owner();
                else
                {
                    for (auto i : f.Owner()->Inputs())
                        back.push_back(i);
                }
        }
        front = back;
        back.clear();
    }
    return nullptr;
}

void TraceGRUPathes(const FunctionPtr &src, string &f_activation, string &g_activation,
    RNNDirection &direction, Variable &initStateH)
{
    std::vector<Variable> inputVars = src->Inputs();
    std::vector<FunctionPtr> pastValueOps, futureValueOps;
    GetDelayOps(inputVars, pastValueOps, futureValueOps);

    // indices here coresponding with CNTK python layer code.
    if (pastValueOps.size() == 1 && futureValueOps.size() == 0)
    {
        direction = RNNDirection::Forward;
        initStateH = pastValueOps[0]->Inputs()[1];
    }
    else if (pastValueOps.size() == 0 && futureValueOps.size() == 1)
    {
        direction = RNNDirection::Backward;
        initStateH = futureValueOps[0]->Inputs()[1];
    }
    else
    {
        CNTK::LogicError("Node %s (%s) is not a valid GRU node", ToLegacyString(ToUTF8(src->Name())).c_str(), ToLegacyString(ToUTF8(src->Uid())).c_str());
    }

    // set up traverse boundary
    std::unordered_set<FunctionPtr> visitedFunctions;
    for (std::vector<Variable>::const_iterator it = inputVars.begin(); it != inputVars.end(); it++)
    {
        visitedFunctions.insert(it->Owner());
    }

    std::vector<std::vector<FunctionPtr>> pathesToPlusSlice;
    std::vector<FunctionPtr> currentPath;

    FunctionPtr gActivation = TraverseGraphFindFirstRNNOp(src->BlockRoot());

    f_activation = "Sigmoid";
    g_activation = MapActivationNameCNTKToONNX(ToLegacyString(ToUTF8(gActivation->OpName())));
}

void TraceRNNPathes(const FunctionPtr &src, string &activation,
    RNNDirection &direction, Variable &initStateH)
{
    std::vector<Variable> inputVars = src->Inputs();
    std::vector<FunctionPtr> pastValueOps, futureValueOps;
    GetDelayOps(inputVars, pastValueOps, futureValueOps);

    // indices here coresponding with CNTK python layer code.
    if (pastValueOps.size() == 1 && futureValueOps.size() == 0)
    {
        direction = RNNDirection::Forward;
        initStateH = pastValueOps[0]->Inputs()[1];
    }
    else if (pastValueOps.size() == 0 && futureValueOps.size() == 1)
    {
        direction = RNNDirection::Backward;
        initStateH = futureValueOps[0]->Inputs()[1];
    }
    else
    {
        CNTK::LogicError("Node %s (%s) is not a valid RNN node", ToLegacyString(ToUTF8(src->Name())).c_str(), ToLegacyString(ToUTF8(src->Uid())).c_str());
    }

    FunctionPtr activationFunction = src->BlockRoot();
    activation = MapActivationNameCNTKToONNX(ToLegacyString(ToUTF8(activationFunction->OpName())));
}

std::vector<FunctionPtr> GetRNNBlocksFromSingleOrBidirectionalRNN(const FunctionPtr src, const std::string &RNNStepOpName)
{
    std::vector<FunctionPtr> rnns;
    if (ToLegacyString(ToUTF8(src->OpName())) == RNNStepOpName)
    {
        rnns.push_back(src);
    }
    else if (src->OpName() == L"Splice") // src is a Splice op with inputs from two LSTM ops.
    {
        for (auto &input : src->Inputs())
        {
            rnns.push_back(input.Owner());
        }
    }
    else
    {
        CNTK::LogicError("An %s op should start with an GRU op (single direction) or a Splice op (bidirectional).", RNNStepOpName.c_str());
    }

    // For single direction RNN,  rnns.size() == 1. For bidirectional RNN, rnns.size() == 2.
    // It is an error otherwise.
    if (rnns.size() == 0 || rnns.size() > 2 ||
        std::any_of(rnns.cbegin(), rnns.cend(), [RNNStepOpName](const FunctionPtr &f) { return ToLegacyString(ToUTF8(f->OpName())) != RNNStepOpName; }))
    {
        CNTK::LogicError("Invalid number of RNN ops to construct an ONNX %s node.", RNNStepOpName.c_str());
    }

    return rnns;
}
